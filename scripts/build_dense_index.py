"""Script de construction de l'index dense vectoriel sémantique.

Caractéristiques :
- Modèle : BAAI/bge-small-en-v1.5 (baseline léger et reproductible sur CPU)
- Encodage par blocs avec checkpoints de sauvegarde intermédiaire (reprise transparente)
- Tri par longueur à l'intérieur de chaque tranche pour optimiser le padding, avec
  réalignement strict sur l'ordre d'origine
- Suivi de la troncature (> 512 tokens) consigné dans metadata.json (passages.jsonl intact)
- Rapport complet : temps mesuré, pic mémoire RAM, taille disque, passages tronqués par type
"""

import argparse
import hashlib
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

# Ajout de la racine du projet
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import psutil
import torch
from tqdm import tqdm
from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer

from src.dense_retriever import (
    DenseRetriever,
    compute_file_sha256,
)
from src.indexing_utils import (
    INDEXING_RULES_VERSION,
    text_for_indexing,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("build_dense_index")


def get_process_memory_mb() -> float:
    """Retourne la mémoire RSS utilisée par le processus en Mo."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)


def get_model_commit_hash(model_name: str) -> str:
    """Tente de résoudre l'empreinte de commit exacte (SHA) du modèle Hugging Face."""
    # 1. Inspection du cache local Hugging Face (rapide, hors-ligne)
    try:
        from huggingface_hub import scan_cache_dir
        for repo in scan_cache_dir().repos:
            if repo.repo_id == model_name:
                for rev in repo.revisions:
                    if rev.commit_hash:
                        return rev.commit_hash
    except Exception:
        pass
    # 2. Requête vers le Hub Hugging Face si le réseau est accessible
    try:
        from huggingface_hub import model_info
        info = model_info(model_name)
        if getattr(info, "sha", None):
            return str(info.sha)
    except Exception:
        pass
    return ""


def parse_args():
    parser = argparse.ArgumentParser(description="Construit l'index dense vectoriel avec SentenceTransformers.")
    parser.add_argument("--passages", type=str, default="data/passages.jsonl", help="Chemin du corpus JSONL.")
    parser.add_argument("--field", type=str, default="content", choices=["content", "contextual_content"], help="Champ à indexer.")
    parser.add_argument("--output", type=str, default="data/index/dense_content", help="Dossier de sortie.")
    parser.add_argument("--model-name", type=str, default="BAAI/bge-small-en-v1.5", help="Nom du modèle Hugging Face.")
    parser.add_argument("--model-revision", type=str, default=None, help="Hash de commit Hugging Face du modèle (détecté automatiquement si omis).")
    parser.add_argument("--query-prefix", type=str, default="Represent this sentence for searching relevant passages: ", help="Préfixe d'instruction pour les requêtes.")
    parser.add_argument("--batch-size", type=int, default=32, help="Taille des batchs d'encodage.")
    parser.add_argument("--checkpoint-every", type=int, default=1000, help="Nombre de passages par bloc de checkpoint.")
    parser.add_argument("--limit", type=int, default=None, help="Nombre maximal de passages à encoder (pour tests).")
    parser.add_argument("--threads", "--num-threads", dest="num_threads", type=int, default=4, help="Nombre de threads PyTorch alloués au CPU (défaut : 4 cœurs physiques).")
    parser.add_argument("--device", type=str, default="cpu", help="Périphérique d'inférence ('cpu' ou 'cuda').")
    return parser.parse_args()


def main():
    args = parse_args()
    torch.set_num_threads(args.num_threads)
    t0_start = time.perf_counter()

    passages_path = Path(args.passages)
    out_dir = Path(args.output)
    ckpt_dir = out_dir / "checkpoints"
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    model_revision = args.model_revision or get_model_commit_hash(args.model_name)
    if not model_revision:
        logger.warning("Impossible de résoudre l'empreinte de commit du modèle %s.", args.model_name)

    print("=" * 65)
    print(f"CONSTRUCTION DE L'INDEX DENSE VECTORIEL : {args.field.upper()}")
    print("=" * 65)
    print(f"Modèle             : {args.model_name} (baseline léger et reproductible sur CPU)")
    if model_revision:
        print(f"Révision du modèle : {model_revision}")
    print(f"Fichier source     : {passages_path}")
    print(f"Dossier de sortie  : {out_dir}")
    print(f"Threads PyTorch    : {args.num_threads} (CPU)")
    print(f"Taille batch       : {args.batch_size}")
    print(f"Taille checkpoint  : {args.checkpoint_every} passages")

    # 1. Chargement des passages
    print(f"\n[1/5] Lecture des passages depuis {passages_path}...")
    with open(passages_path, "r", encoding="utf-8") as f:
        all_passages = [json.loads(line) for line in f]

    if args.limit:
        all_passages = all_passages[:args.limit]
        print(f"  -> Mode test activé : limitation à {len(all_passages)} passages.")

    total_passages = len(all_passages)
    passage_ids = [p["passage_id"] for p in all_passages]
    print(f"  -> Total de passages à indexer : {total_passages}")

    # Empreinte SHA-256 du fichier complet
    corpus_sha256 = compute_file_sha256(passages_path)
    print(f"  -> Empreinte SHA-256 du corpus : {corpus_sha256}")

    # 2. Analyse des longueurs de tokens & Troncature (> 512)
    print(f"\n[2/5] Analyse des tokens ({args.model_name}) pour le suivi de la troncature...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    passage_tracking: Dict[str, Dict[str, Any]] = {}
    type_stats = {
        "text": {"total": 0, "truncated": 0},
        "table": {"total": 0, "truncated": 0},
        "figure": {"total": 0, "truncated": 0},
    }

    texts_to_encode: List[str] = []
    for p in all_passages:
        pid = p["passage_id"]
        ptype = p.get("type", "text")
        raw_text = text_for_indexing(p, field=args.field)
        texts_to_encode.append(raw_text)

        # Calcul exact des tokens avec les tokens spéciaux [CLS] et [SEP]
        n_tok = len(tokenizer.encode(raw_text, truncation=False))
        is_trunc = n_tok > 512

        passage_tracking[pid] = {
            "n_tokens": n_tok,
            "truncated": is_trunc,
        }

        if ptype in type_stats:
            type_stats[ptype]["total"] += 1
            if is_trunc:
                type_stats[ptype]["truncated"] += 1

    total_truncated = sum(s["truncated"] for s in type_stats.values())
    print("  Résultats de la troncature (> 512 tokens) :")
    for ptype, s in type_stats.items():
        if s["total"] > 0:
            pct = s["truncated"] / s["total"] * 100
            print(f"    - Type {ptype:<6} : {s['truncated']:4d} / {s['total']:5d} ({pct:5.2f}%)")
    print(f"    - TOTAL GLOBAL  : {total_truncated:4d} / {total_passages:5d} ({total_truncated / total_passages * 100:5.2f}%)")

    # 3. Chargement du modèle SentenceTransformer
    print(f"\n[3/5] Chargement du modèle d'embedding sur {args.device}...")
    model = SentenceTransformer(args.model_name, device=args.device)

    # 4. Encodage par blocs de checkpoints avec reprise
    print(f"\n[4/5] Encodage vectoriel par tranches de {args.checkpoint_every} passages...")
    checkpoint_files: List[Path] = []
    chunk_size = args.checkpoint_every
    num_chunks = (total_passages + chunk_size - 1) // chunk_size

    for chunk_idx in range(num_chunks):
        c_start = chunk_idx * chunk_size
        c_end = min(c_start + chunk_size, total_passages)
        ckpt_file = ckpt_dir / f"chunk_{c_start:06d}_{c_end:06d}.npy"
        ckpt_meta_file = ckpt_dir / f"chunk_{c_start:06d}_{c_end:06d}.meta.json"
        checkpoint_files.append(ckpt_file)

        if ckpt_file.exists() and ckpt_meta_file.exists():
            try:
                with open(ckpt_meta_file, "r", encoding="utf-8") as f:
                    ckpt_meta = json.load(f)
                if ckpt_meta.get("indexing_rules_version") == INDEXING_RULES_VERSION:
                    print(f"  [Tranche {chunk_idx + 1}/{num_chunks}] Déjà encodée ({c_start} à {c_end}, version {INDEXING_RULES_VERSION}) -> Chargement du checkpoint.")
                    continue
                else:
                    print(f"  [Tranche {chunk_idx + 1}/{num_chunks}] Version des règles expirée ({ckpt_meta.get('indexing_rules_version')} != {INDEXING_RULES_VERSION}) -> Recalcul du bloc.")
            except Exception:
                pass

        print(f"  [Tranche {chunk_idx + 1}/{num_chunks}] Encodage des passages {c_start} à {c_end}...")
        chunk_texts = texts_to_encode[c_start:c_end]

        # Tri local par longueur pour optimiser le padding intra-batch
        indexed_texts = list(enumerate(chunk_texts))
        indexed_texts.sort(key=lambda item: len(item[1]))
        sorted_texts = [item[1] for item in indexed_texts]
        original_indices = [item[0] for item in indexed_texts]

        # Encodage par batchs
        chunk_embs_sorted = model.encode(
            sorted_texts,
            batch_size=args.batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,
            device=args.device,
        )

        # Réalignement sur l'ordre d'origine
        chunk_embs = np.empty_like(chunk_embs_sorted, dtype=np.float32)
        for sorted_pos, orig_pos in enumerate(original_indices):
            chunk_embs[orig_pos] = chunk_embs_sorted[sorted_pos]

        # Sauvegarde du checkpoint intermédiaire et de ses métadonnées de version
        np.save(ckpt_file, chunk_embs)
        with open(ckpt_meta_file, "w", encoding="utf-8") as f:
            json.dump({
                "indexing_rules_version": INDEXING_RULES_VERSION,
                "c_start": c_start,
                "c_end": c_end,
                "shape": list(chunk_embs.shape),
            }, f, indent=2)
        print(f"  -> Checkpoint sauvegardé : {ckpt_file.name} ({chunk_embs.shape})")

    # 5. Assemblage final et enregistrement des métadonnées
    print("\n[5/5] Assemblage des checkpoints en matrice finale...")
    all_chunks_list = [np.load(f).astype(np.float32) for f in checkpoint_files]
    final_embeddings = np.vstack(all_chunks_list)

    if len(final_embeddings) != total_passages:
        raise ValueError(
            f"Erreur d'assemblage : {len(final_embeddings)} vecteurs assemblés "
            f"contre {total_passages} passages attendus."
        )

    # Sauvegarde des matrices et métadonnées
    final_emb_path = out_dir / "embeddings.npy"
    np.save(final_emb_path, final_embeddings)

    ids_path = out_dir / "passage_ids.json"
    with open(ids_path, "w", encoding="utf-8") as f:
        json.dump(passage_ids, f, ensure_ascii=False)

    total_time = time.perf_counter() - t0_start
    peak_ram_mb = get_process_memory_mb()
    file_size_mb = final_emb_path.stat().st_size / (1024 * 1024)

    metadata = {
        "model_name": args.model_name,
        "model_revision": model_revision,
        "model_description": "baseline léger et reproductible sur CPU",
        "dimension": int(final_embeddings.shape[1]),
        "field": args.field,
        "query_prefix": args.query_prefix,
        "indexing_rules_version": INDEXING_RULES_VERSION,
        "n_passages": total_passages,
        "passages_sha256": corpus_sha256,
        "indexing_time_seconds": round(total_time, 2),
        "peak_ram_mb": round(peak_ram_mb, 2),
        "embeddings_file_size_mb": round(file_size_mb, 2),
        "truncated_summary": {
            ptype: {
                "total": s["total"],
                "truncated": s["truncated"],
                "percentage": round(s["truncated"] / s["total"] * 100, 2) if s["total"] > 0 else 0.0,
            }
            for ptype, s in type_stats.items()
        },
        "passage_tracking": passage_tracking,
    }

    meta_path = out_dir / "metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    meta_size_mb = meta_path.stat().st_size / (1024 * 1024)

    print("\n" + "=" * 65)
    print("RAPPORT D'INDEXATION DENSE VECTORIELLE")
    print("=" * 65)
    print(f"Modèle                 : {args.model_name} (révision: {model_revision or 'inconnue'})")
    print(f"Nombre de passages     : {total_passages}")
    print(f"Forme de la matrice    : {final_embeddings.shape} (float32)")
    print(f"Taille embeddings.npy  : {file_size_mb:.2f} Mo")
    print(f"Taille metadata.json   : {meta_size_mb:.2f} Mo")
    print(f"Pic mémoire RAM        : {peak_ram_mb:.1f} Mo")
    print(f"Temps total d'exécution: {total_time:.1f} s ({total_time / 60:.1f} minutes)")
    print(f"Passages tronqués      : {total_truncated} / {total_passages} ({total_truncated / total_passages * 100:.2f}%)")
    print("=" * 65)
    print(f"Index prêt dans {out_dir}")


if __name__ == "__main__":
    main()
