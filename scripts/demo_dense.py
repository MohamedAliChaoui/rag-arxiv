"""Script de démonstration et d'évaluation descriptive du retriever dense (Étape 4).

Exécute :
1. Recherche sur 6 requêtes types et affichage détaillé du top-5 (score, rank, section, aperçu).
2. Recouvrement descriptif dense vs BM25 (passages communs dans le top-5, indice de Jaccard).
3. Test de cohérence d'auto-récupération sur 200 passages (graine 42, requête = 15 premiers mots).
"""

import argparse
import json
import logging
import random
import sys
from pathlib import Path
from typing import Dict, List, Set

# Racine du projet
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.dense_retriever import DenseRetriever
from src.bm25_retriever import BM25Retriever

logger = logging.getLogger("demo_dense")

TEST_QUERIES = [
    "chunk size and overlap in RAG",
    "when should the model decide to retrieve",
    "late chunking embedding similarity",
    "benchmarks for retrieval evaluation",
    "medical clinical guidelines retrieval",
    "protein folding prediction",  # Requête hors-domaine / test de discrimination
]


def parse_args():
    parser = argparse.ArgumentParser(description="Démonstration du retriever dense et comparaison avec BM25.")
    parser.add_argument("--dense-index", type=str, default="data/index/dense_content", help="Dossier de l'index dense.")
    parser.add_argument("--bm25-index", type=str, default="data/index/bm25", help="Dossier de l'index BM25.")
    parser.add_argument("--passages", type=str, default="data/passages.jsonl", help="Fichier des passages JSONL.")
    parser.add_argument("--k", type=int, default=5, help="Nombre de résultats par requête.")
    parser.add_argument("--device", type=str, default="cpu", help="Périphérique ('cpu' ou 'cuda').")
    return parser.parse_args()


def display_results(query: str, results, retriever_name: str):
    """Affiche le top-k des résultats de manière lisible."""
    print(f"\n--- {retriever_name.upper()} : '{query}' ---")
    for r in results:
        pid = r.passage_id
        score = r.score
        rank = r.rank
        meta = r.metadata or {}
        doc_id = meta.get("doc_id", pid.split("_")[0])
        sec = meta.get("section_path", "N/A")
        ptype = meta.get("type", "text")
        content = meta.get("content", "")
        preview = (content[:120] + "...") if len(content) > 120 else content
        preview = preview.replace("\n", " ")

        print(f"  Rang {rank} | Score: {score:7.4f} | [{ptype.upper()}] {pid} (Doc: {doc_id})")
        print(f"         Section: {sec}")
        print(f"         Extrait: {preview}")


def main():
    args = parse_args()
    dense_path = Path(args.dense_index)
    passages_path = Path(args.passages)
    bm25_path = Path(args.bm25_index)

    if not (dense_path / "embeddings.npy").exists():
        print(f"\n[ATTENTION] L'index dense introuvable dans {dense_path}.")
        print("Veuillez construire l'index au préalable avec :")
        print("  python scripts/build_dense_index.py --field content")
        sys.exit(1)

    print("=" * 70)
    print("DÉMONSTRATION DU RETRIEVAL DENSE VECTORIEL (BGE-small-en-v1.5)")
    print("=" * 70)
    print(f"Index dense : {dense_path}")
    print(f"Passages    : {passages_path}")

    print("\nChargement de l'index dense...")
    dense_retriever = DenseRetriever.load(
        index_dir=dense_path,
        passages_path=passages_path,
        verify_fingerprint=True,
        device=args.device,
    )
    print(f"Index dense chargé : {len(dense_retriever.embeddings)} vecteurs (dim {dense_retriever.embeddings.shape[1]}).")

    # Chargement de BM25 si disponible pour comparaison de recouvrement
    bm25_retriever = None
    if (bm25_path / "metadata.json").exists():
        print(f"Chargement de l'index BM25 depuis {bm25_path} pour comparaison...")
        try:
            bm25_retriever = BM25Retriever.load(
                index_dir=bm25_path,
                passages_path=passages_path,
                verify_fingerprint=True,
            )
        except Exception as e:
            print(f"Avertissement : échec du chargement de BM25 ({e}).")

    # 1. Évaluation des 6 requêtes types
    print("\n" + "=" * 70)
    print("1. RÉSULTATS DES 6 REQUÊTES TYPES (TOP-5)")
    print("=" * 70)

    overlap_stats = []

    for idx, query in enumerate(TEST_QUERIES, start=1):
        print(f"\n=======================================================")
        print(f"REQUÊTE {idx}/{len(TEST_QUERIES)} : \"{query}\"")
        print(f"=======================================================")

        dense_res = dense_retriever.search(query, k=args.k)
        display_results(query, dense_res, "Dense (BGE-small)")

        if bm25_retriever:
            bm25_res = bm25_retriever.search(query, k=args.k)
            display_results(query, bm25_res, "BM25 (Lucene)")

            dense_pids = {r.passage_id for r in dense_res}
            bm25_pids = {r.passage_id for r in bm25_res}
            common = dense_pids & bm25_pids
            union = dense_pids | bm25_pids
            jaccard = len(common) / len(union) if union else 0.0

            overlap_stats.append({
                "query": query,
                "common_count": len(common),
                "jaccard": jaccard,
                "common_pids": list(common),
            })

            print(f"\n  [Recouvrement Top-{args.k}] : {len(common)}/{args.k} passages en commun (Jaccard: {jaccard:.2f})")
            if common:
                print(f"    Passages partagés : {', '.join(common)}")

    # 2. Bilan descriptif du recouvrement Dense vs BM25
    if overlap_stats:
        print("\n" + "=" * 70)
        print("2. BILAN DU RECOUVREMENT DESCRIPTIF DENSE VS BM25 (SANS JUGEMENT QUALITATIF)")
        print("=" * 70)
        avg_common = sum(s["common_count"] for s in overlap_stats) / len(overlap_stats)
        avg_jaccard = sum(s["jaccard"] for s in overlap_stats) / len(overlap_stats)
        for s in overlap_stats:
            print(f"  - \"{s['query'][:40]:<40}\" : {s['common_count']}/{args.k} communs (Jaccard {s['jaccard']:.2f})")
        print(f"\n  Recouvrement moyen : {avg_common:.1f}/{args.k} passages ({avg_jaccard:.2f} Jaccard)")
        print("  Note méthodologique : Aucune métrique de performance qualitative (NDCG, Recall)")
        print("  n'est calculée à ce stade. L'évaluation formelle sera réalisée à l'étape 8.")

    # 3. Test de cohérence d'auto-récupération (200 passages, graine 42)
    print("\n" + "=" * 70)
    print("3. TEST DE COHÉRENCE D'AUTO-RÉCUPÉRATION (200 PASSAGES, GRAINE 42)")
    print("=" * 70)
    all_passages = dense_retriever.passages
    random.seed(42)
    sample_passages = random.sample(all_passages, 200)

    top1_hits = 0
    top5_hits = 0

    print("Encodage des 200 pseudo-requêtes (15 premiers mots) et recherche vectorielle...")
    for idx, p in enumerate(sample_passages, start=1):
        target_pid = p["passage_id"]
        words = p.get("content", "").split()
        if not words:
            continue
        pseudo_query = " ".join(words[:15])

        results = dense_retriever.search(pseudo_query, k=5)
        hit_pids = [r.passage_id for r in results]

        if hit_pids and hit_pids[0] == target_pid:
            top1_hits += 1
        if target_pid in hit_pids:
            top5_hits += 1

    pct_top1 = top1_hits / len(sample_passages) * 100
    pct_top5 = top5_hits / len(sample_passages) * 100

    print(f"Résultats d'auto-récupération dense sur 200 passages :")
    print(f"  - Top-1 hit rate : {top1_hits:3d} / 200 ({pct_top1:5.1f}%)")
    print(f"  - Top-5 hit rate : {top5_hits:3d} / 200 ({pct_top5:5.1f}%)")
    print("=" * 70)


if __name__ == "__main__":
    main()
