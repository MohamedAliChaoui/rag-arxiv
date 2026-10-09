"""Module de retrieval dense vectoriel sémantique.

Utilise SentenceTransformers (par défaut BAAI/bge-small-en-v1.5) et une recherche
vectorielle par produit scalaire normalisé L2 sous NumPy (float32).
Respecte le contrat d'interface BaseRetriever.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np

from src.indexing_utils import INDEXING_RULES_VERSION, text_for_indexing
from src.retriever_base import BaseRetriever, SearchResult

logger = logging.getLogger(__name__)


def compute_file_sha256(filepath: Union[str, Path]) -> str:
    """Calcule l'empreinte SHA-256 d'un fichier."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


class DenseRetriever(BaseRetriever):
    """Moteur de recherche sémantique dense reposant sur NumPy float32 et SentenceTransformers."""

    DEFAULT_MODEL_NAME = "BAAI/bge-small-en-v1.5"
    DEFAULT_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        query_prefix: Optional[str] = DEFAULT_QUERY_PREFIX,
        field: str = "content",
        embeddings: Optional[np.ndarray] = None,
        passage_ids: Optional[List[str]] = None,
        passages: Optional[List[Dict[str, Any]]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        device: str = "cpu",
    ) -> None:
        """Initialise le retriever dense."""
        self.model_name = model_name
        self.query_prefix = query_prefix or ""
        self.field = field
        self.device = device
        self._model = None  # Chargé à la demande pour l'inférence de requêtes

        self.embeddings = embeddings  # Matrice (N, D) float32 normalisée L2
        self.passage_ids = passage_ids or []
        self.passages = passages or []
        self.metadata = metadata or {}
        self.model_revision = self.metadata.get("model_revision")

        # Table d'accès rapide id -> passage
        self._passage_map: Dict[str, Dict[str, Any]] = {
            p["passage_id"]: p for p in self.passages if "passage_id" in p
        }

    @property
    def model(self):
        """Charge le modèle SentenceTransformer de manière paresseuse (lazy loading)."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            logger.info("Chargement du modèle dense %s sur %s...", self.model_name, self.device)
            self._model = SentenceTransformer(self.model_name, device=self.device)
        return self._model

    def encode_queries(self, queries: List[str]) -> np.ndarray:
        """Encode une liste de requêtes en appliquant le préfixe d'instruction configuré."""
        prefixed_queries = [
            f"{self.query_prefix}{q}" if self.query_prefix else q
            for q in queries
        ]
        embeddings = self.model.encode(
            prefixed_queries,
            batch_size=32,
            show_progress_bar=False,
            normalize_embeddings=True,
            device=self.device,
        )
        return np.asarray(embeddings, dtype=np.float32)

    def search(self, query: str, k: int = 5) -> List[SearchResult]:
        """Recherche sémantique par produit scalaire (similarité cosinus)."""
        if self.embeddings is None or len(self.passage_ids) == 0:
            raise ValueError("L'index dense n'est pas chargé ou est vide.")

        k = max(1, min(k, len(self.passage_ids)))

        # 1. Encodage de la requête
        query_vec = self.encode_queries([query])[0]  # vecteur (D,) normalisé L2

        # 2. Produit scalaire Q . D^T (équivalent cosinus puisque tous les vecteurs sont normalisés)
        scores = np.dot(self.embeddings, query_vec)

        # 3. Sélection des k meilleurs scores
        if k < len(scores):
            # argpartition pour sélectionner rapidement les indices du top-k
            top_k_unsorted = np.argpartition(scores, -k)[-k:]
            # Tri précis descendant parmi les k sélectionnés
            top_k_indices = top_k_unsorted[np.argsort(-scores[top_k_unsorted])]
        else:
            top_k_indices = np.argsort(-scores)

        results: List[SearchResult] = []
        for rank, idx in enumerate(top_k_indices, start=1):
            pid = self.passage_ids[idx]
            score_val = float(scores[idx])
            p_data = self._passage_map.get(pid, {})
            results.append(
                SearchResult(
                    passage_id=pid,
                    score=score_val,
                    rank=rank,
                    metadata=p_data,
                )
            )

        return results

    def save(self, index_dir: Union[str, Path]) -> None:
        """Sauvegarde les embeddings, la liste d'IDs et les métadonnées sur disque."""
        out_path = Path(index_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        if self.embeddings is None:
            raise ValueError("Aucun embedding à sauvegarder.")

        # Matrice NumPy float32
        np.save(out_path / "embeddings.npy", self.embeddings.astype(np.float32))

        # Liste d'identifiants alignée
        with open(out_path / "passage_ids.json", "w", encoding="utf-8") as f:
            json.dump(self.passage_ids, f, ensure_ascii=False, indent=2)

        # Métadonnées
        self.metadata["indexing_rules_version"] = INDEXING_RULES_VERSION
        with open(out_path / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, ensure_ascii=False, indent=2)

        logger.info(
            "Index dense sauvegardé avec succès dans %s (%d passages, dim %d).",
            out_path,
            len(self.passage_ids),
            self.embeddings.shape[1],
        )

    @classmethod
    def load(
        cls,
        index_dir: Union[str, Path],
        passages_path: Optional[Union[str, Path]] = None,
        verify_fingerprint: bool = True,
        device: str = "cpu",
    ) -> "DenseRetriever":
        """Charge un index dense depuis le disque avec vérification d'intégrité SHA-256."""
        in_path = Path(index_dir)
        emb_file = in_path / "embeddings.npy"
        ids_file = in_path / "passage_ids.json"
        meta_file = in_path / "metadata.json"

        if not emb_file.exists() or not ids_file.exists():
            raise FileNotFoundError(
                f"Fichiers d'index dense introuvables dans {in_path} "
                f"(requis : embeddings.npy, passage_ids.json)"
            )

        embeddings = np.load(emb_file).astype(np.float32)

        with open(ids_file, "r", encoding="utf-8") as f:
            passage_ids = json.load(f)

        metadata: Dict[str, Any] = {}
        if meta_file.exists():
            with open(meta_file, "r", encoding="utf-8") as f:
                metadata = json.load(f)

        # Contrôle de version des règles d'indexation
        saved_rules_version = metadata.get("indexing_rules_version")
        if saved_rules_version and saved_rules_version != INDEXING_RULES_VERSION:
            raise ValueError(
                f"Incompatibilité de version des règles d'indexation : "
                f"l'index utilise la version '{saved_rules_version}', "
                f"mais le code requiert '{INDEXING_RULES_VERSION}'."
            )

        if len(embeddings) != len(passage_ids):
            raise ValueError(
                f"Désalignement : {len(embeddings)} vecteurs contre {len(passage_ids)} passage_ids."
            )

        # Chargement des passages si le chemin est fourni
        passages = []
        if passages_path is not None:
            p_path = Path(passages_path)
            if p_path.exists():
                # Contrôle d'empreinte SHA-256
                saved_sha = metadata.get("passages_sha256")
                if saved_sha and verify_fingerprint:
                    current_sha = compute_file_sha256(p_path)
                    if current_sha != saved_sha:
                        raise ValueError(
                            f"L'empreinte SHA-256 du corpus ({current_sha}) ne correspond pas "
                            f"à celle enregistrée lors de l'indexation dense ({saved_sha})."
                        )

                with open(p_path, "r", encoding="utf-8") as f:
                    passages = [json.loads(line) for line in f]

        model_name = metadata.get("model_name", cls.DEFAULT_MODEL_NAME)
        query_prefix = metadata.get("query_prefix", cls.DEFAULT_QUERY_PREFIX)
        field = metadata.get("field", "content")

        return cls(
            model_name=model_name,
            query_prefix=query_prefix,
            field=field,
            embeddings=embeddings,
            passage_ids=passage_ids,
            passages=passages,
            metadata=metadata,
            device=device,
        )
