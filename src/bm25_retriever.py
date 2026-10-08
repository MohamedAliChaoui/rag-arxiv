"""
Module d'indexation et de recherche lexicale BM25 fondé sur la bibliothèque bm25s.

Variante d'implémentation (bm25s method='lucene') :
---------------------------------------------------
Le moteur bm25s utilise par défaut la variante Lucene d'Okapi BM25
(décrite dans Kamphuis et al., ECIR 2020) :

1. Inverse Document Frequency (IDF) :
   IDF(t) = ln(1 + (N - df_t + 0.5) / (df_t + 0.5))
   L'ajout du terme '+ 1' à l'intérieur du logarithme garantit que l'IDF
   reste strictement positif, même pour les termes très fréquents (df_t > N / 2),
   évitant les poids négatifs de la formule originale de Robertson.

2. Term Frequency Component (TFC) :
   TFC(t, D) = tf_{t,D} / (tf_{t,D} + k1 * (1 - b + b * (|D| / avgdl)))
   Le facteur multiplicatif constant (k1 + 1) au numérateur est omis dans Lucene,
   car il s'agit d'une constante d'échelle uniforme qui ne modifie pas le rang des documents.

Score total :
   score(D, Q) = sum_{t in Q} IDF(t) * TFC(t, D)
"""

from datetime import datetime, timezone
import hashlib
import json
import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import bm25s

from src.retriever_base import BaseRetriever, SearchResult
from src.tokenizer import ScientificTokenizer, TokenizerConfig

logger = logging.getLogger(__name__)


def compute_file_sha256(filepath: str) -> str:
    """Calcule l'empreinte cryptographique SHA-256 d'un fichier."""
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


class BM25Retriever(BaseRetriever):
    """
    Retriever lexical BM25 exploitant bm25s avec tokenisation scientifique personnalisée
    et persistance sérialisée sur disque avec contrôle d'intégrité par empreinte SHA-256.
    """

    def __init__(
        self,
        indexed_field: str = "content",
        indexed_types: Optional[List[str]] = None,
        min_words: int = 0,
        tokenizer_config: Optional[TokenizerConfig] = None,
        k1: float = 1.5,
        b: float = 0.75,
        method: str = "lucene",
    ):
        self.indexed_field = indexed_field  # 'content' ou 'contextual_content'
        self.indexed_types = indexed_types or ["text", "table", "figure"]
        self.min_words = min_words
        self.tokenizer_config = tokenizer_config or TokenizerConfig()
        self.tokenizer = ScientificTokenizer(self.tokenizer_config)

        self.k1 = k1
        self.b = b
        self.method = method

        self.bm25: Optional[bm25s.BM25] = None
        self.passage_ids: List[str] = []
        self.passages_metadata: Dict[str, Dict[str, Any]] = {}
        self.metadata_info: Dict[str, Any] = {}

    def build_index(
        self,
        passages_path: str = "data/passages.jsonl",
        show_progress: bool = True,
    ) -> Dict[str, Any]:
        """
        Construit l'index inversé BM25 à partir du fichier passages.jsonl.
        """
        if not os.path.exists(passages_path):
            raise FileNotFoundError(f"Fichier de passages introuvable : {passages_path}")

        file_sha256 = compute_file_sha256(passages_path)
        corpus_tokens: List[List[str]] = []
        passage_ids: List[str] = []
        passages_metadata: Dict[str, Dict[str, Any]] = {}

        with open(passages_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                p = json.loads(line)

                # Filtrage par type
                if p.get("type") not in self.indexed_types:
                    continue

                # Filtrage par longueur
                if p.get("word_count", 0) < self.min_words:
                    continue

                pid = p["passage_id"]
                raw_text = p.get(self.indexed_field, "")
                if not raw_text:
                    continue

                tokens = self.tokenizer.tokenize(raw_text)
                if not tokens:
                    continue

                corpus_tokens.append(tokens)
                passage_ids.append(pid)
                passages_metadata[pid] = {
                    "arxiv_id": p.get("arxiv_id", ""),
                    "title": p.get("title", ""),
                    "section_id": p.get("section_id", ""),
                    "section_path": p.get("section_path", ""),
                    "type": p.get("type", ""),
                    "word_count": p.get("word_count", 0),
                    "content_snippet": p.get("content", "")[:250],
                }

        # Initialisation et calcul de l'index via bm25s
        self.bm25 = bm25s.BM25(
            k1=self.k1,
            b=self.b,
            method=self.method,
            corpus=passage_ids,
        )
        self.bm25.index(corpus_tokens, show_progress=show_progress)

        self.passage_ids = passage_ids
        self.passages_metadata = passages_metadata
        self.metadata_info = {
            "indexed_field": self.indexed_field,
            "indexed_types": self.indexed_types,
            "min_words": self.min_words,
            "num_passages": len(passage_ids),
            "passages_file": passages_path,
            "passages_sha256": file_sha256,
            "tokenizer_config": self.tokenizer_config.to_dict(),
            "bm25_params": {
                "k1": self.k1,
                "b": self.b,
                "method": self.method,
            },
            "created_at": datetime.now(timezone.utc).isoformat(),
        }

        return self.metadata_info

    def save(self, index_dir: str = "data/index/bm25") -> None:
        """
        Sauvegarde l'index bm25s et le fichier de métadonnées associé sur disque.
        """
        if self.bm25 is None:
            raise ValueError("Aucun index n'a été construit ou chargé.")

        os.makedirs(index_dir, exist_ok=True)
        # 1. Sauvegarde des structures sparse bm25s
        self.bm25.save(index_dir, show_progress=False)

        # 2. Sauvegarde des métadonnées du retriever et de l'empreinte source
        meta_filepath = os.path.join(index_dir, "metadata.json")
        full_metadata = {
            **self.metadata_info,
            "passage_ids": self.passage_ids,
            "passages_metadata": self.passages_metadata,
        }
        with open(meta_filepath, "w", encoding="utf-8") as f:
            json.dump(full_metadata, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(
        cls,
        index_dir: str = "data/index/bm25",
        verify_fingerprint: bool = True,
        strict_fingerprint: bool = False,
    ) -> "BM25Retriever":
        """
        Charge un index BM25 précalculé depuis le disque avec vérification d'empreinte.
        """
        meta_filepath = os.path.join(index_dir, "metadata.json")
        if not os.path.exists(meta_filepath):
            raise FileNotFoundError(f"Métadonnées introuvables : {meta_filepath}")

        with open(meta_filepath, "r", encoding="utf-8") as f:
            meta = json.load(f)

        t_config = TokenizerConfig.from_dict(meta.get("tokenizer_config", {}))
        bm25_params = meta.get("bm25_params", {})

        retriever = cls(
            indexed_field=meta.get("indexed_field", "content"),
            indexed_types=meta.get("indexed_types", ["text", "table", "figure"]),
            min_words=meta.get("min_words", 0),
            tokenizer_config=t_config,
            k1=bm25_params.get("k1", 1.5),
            b=bm25_params.get("b", 0.75),
            method=bm25_params.get("method", "lucene"),
        )

        retriever.passage_ids = meta.get("passage_ids", [])
        retriever.passages_metadata = meta.get("passages_metadata", {})
        retriever.metadata_info = {k: v for k, v in meta.items() if k not in ("passage_ids", "passages_metadata")}

        # Contrôle d'empreinte SHA-256 de passages.jsonl
        passages_path = meta.get("passages_file", "data/passages.jsonl")
        expected_hash = meta.get("passages_sha256")
        if verify_fingerprint and os.path.exists(passages_path) and expected_hash:
            current_hash = compute_file_sha256(passages_path)
            if current_hash != expected_hash:
                msg = (
                    f"Alerte : L'empreinte de {passages_path} a changé depuis la création de l'index.\n"
                    f"Attendu : {expected_hash}\nActuel  : {current_hash}"
                )
                if strict_fingerprint:
                    raise ValueError(f"Désynchronisation critique de l'index : {msg}")
                else:
                    logger.warning(msg)

        # Chargement des matrices bm25s
        retriever.bm25 = bm25s.BM25.load(index_dir, load_corpus=False)
        return retriever

    def search(self, query: str, k: int = 10) -> List[SearchResult]:
        """
        Recherche les k passages les plus pertinents pour une requête.
        Retourne une liste de SearchResult(passage_id, score, rank, metadata).
        """
        if self.bm25 is None:
            raise ValueError("L'index BM25 n'est pas initialisé.")

        if not query or not query.strip():
            return []

        query_tokens = self.tokenizer.tokenize(query)
        if not query_tokens:
            return []

        # Récupération des scores bruts pour tous les documents
        scores_arr = self.bm25.get_scores(query_tokens)
        if len(scores_arr) == 0:
            return []

        # Sélection des k meilleurs indices
        top_k = min(k, len(scores_arr))
        # Utilisation d'argpartition pour les performances
        if len(scores_arr) > top_k:
            partition_idx = len(scores_arr) - top_k
            import numpy as np
            top_indices = np.argpartition(scores_arr, partition_idx)[partition_idx:]
            top_indices = top_indices[np.argsort(scores_arr[top_indices])[::-1]]
        else:
            import numpy as np
            top_indices = np.argsort(scores_arr)[::-1]

        results: List[SearchResult] = []
        rank = 1
        for idx in top_indices:
            score = float(scores_arr[idx])
            if score <= 0.0 and rank > 1:
                # Arrêt si le score tombe à 0 ou négatif
                break
            pid = self.passage_ids[idx]
            results.append(
                SearchResult(
                    passage_id=pid,
                    score=round(score, 4),
                    rank=rank,
                    metadata=self.passages_metadata.get(pid),
                )
            )
            rank += 1

        return results
