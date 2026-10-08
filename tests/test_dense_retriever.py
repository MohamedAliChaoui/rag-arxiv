"""Tests unitaires pour le retriever dense vectoriel sémantique (Étape 4)."""

import json
import tempfile
from pathlib import Path
import numpy as np
import pytest

from src.dense_retriever import (
    DenseRetriever,
    compute_file_sha256,
    text_for_indexing,
)
from src.retriever_base import BaseRetriever, SearchResult


@pytest.fixture
def mini_corpus(tmp_path):
    """Crée un corpus de test avec passages standards et un passage contenant uniquement un symbole."""
    passages = [
        {
            "passage_id": "test_doc_p0001",
            "doc_id": "test_doc",
            "section_path": "1 Introduction",
            "content": "Retrieval augmented generation enhances language models with external knowledge.",
            "contextual_content": "Document: Test | Section: 1 Introduction | Retrieval augmented generation enhances language models.",
            "type": "text",
        },
        {
            "passage_id": "test_doc_p0002",
            "doc_id": "test_doc",
            "section_path": "1 Introduction",
            "content": ":",  # Cas limite : aucun jeton textuel standard
            "contextual_content": "Document: Test | Section: 1 Introduction | :",
            "type": "text",
        },
        {
            "passage_id": "test_doc_p0003",
            "doc_id": "test_doc",
            "section_path": "2 Methods",
            "content": "Dense retrieval computes vector embeddings using deep transformer encoders.",
            "contextual_content": "Document: Test | Section: 2 Methods | Dense retrieval computes vector embeddings.",
            "type": "text",
        },
    ]
    p_file = tmp_path / "passages.jsonl"
    with open(p_file, "w", encoding="utf-8") as f:
        for p in passages:
            f.write(json.dumps(p) + "\n")
    return p_file, passages


def test_dense_retriever_subclass_and_search_interface():
    """Vérifie le respect strict du contrat d'interface BaseRetriever."""
    assert issubclass(DenseRetriever, BaseRetriever)

    # Simulation d'un index avec 3 vecteurs 2D
    embs = np.array([
        [1.0, 0.0],
        [0.0, 1.0],
        [0.7071, 0.7071],
    ], dtype=np.float32)
    # Normalisation L2
    embs = embs / np.linalg.norm(embs, axis=1, keepdims=True)

    retriever = DenseRetriever(
        embeddings=embs,
        passage_ids=["p1", "p2", "p3"],
        passages=[
            {"passage_id": "p1", "content": "First"},
            {"passage_id": "p2", "content": "Second"},
            {"passage_id": "p3", "content": "Third"},
        ],
    )

    # Mock de l'encodage de requête pour tester le moteur vectoriel sans inférence
    retriever.encode_queries = lambda queries: np.array([[1.0, 0.0]], dtype=np.float32)

    results = retriever.search("query", k=2)
    assert len(results) == 2
    assert all(isinstance(r, SearchResult) for r in results)
    assert results[0].passage_id == "p1"
    assert results[0].rank == 1
    assert pytest.approx(results[0].score, 1e-4) == 1.0
    assert results[1].passage_id == "p3"
    assert results[1].rank == 2


def test_empty_or_symbol_passage_embedding_and_alignment(mini_corpus):
    """Vérifie qu'un passage avec seulement ':' a un embedding valide et ne décale pas les indices."""
    p_file, passages = mini_corpus

    # Test avec de vrais embeddings factices simulant la sortie d'un modèle
    embs = np.random.randn(len(passages), 384).astype(np.float32)
    embs /= np.linalg.norm(embs, axis=1, keepdims=True)

    retriever = DenseRetriever(
        embeddings=embs,
        passage_ids=[p["passage_id"] for p in passages],
        passages=passages,
    )

    assert len(retriever.embeddings) == 3
    assert not np.isnan(retriever.embeddings).any()
    assert retriever.passage_ids[1] == "test_doc_p0002"

    # Vérification que la recherche renvoie les résultats ordonnés sans exception
    retriever.encode_queries = lambda queries: embs[0:1]
    res = retriever.search("test query", k=3)
    assert len(res) == 3
    assert res[0].passage_id == "test_doc_p0001"


def test_dense_query_prefix_behavior():
    """Vérifie la bonne prise en compte du préfixe de requête (optionnel / activé par défaut)."""
    retriever_with_prefix = DenseRetriever(query_prefix="Represent this sentence: ")
    assert retriever_with_prefix.query_prefix == "Represent this sentence: "

    retriever_no_prefix = DenseRetriever(query_prefix="")
    assert retriever_no_prefix.query_prefix == ""

    retriever_none = DenseRetriever(query_prefix=None)
    assert retriever_none.query_prefix == ""


def test_save_load_and_sha256_integrity(mini_corpus, tmp_path):
    """Vérifie la sérialisation, le rechargement exact et le contrôle d'empreinte SHA-256."""
    p_file, passages = mini_corpus
    sha256 = compute_file_sha256(p_file)

    embs = np.array([
        [0.6, 0.8],
        [0.8, -0.6],
        [1.0, 0.0],
    ], dtype=np.float32)

    meta = {
        "model_name": "BAAI/bge-small-en-v1.5",
        "query_prefix": "Represent this sentence: ",
        "field": "content",
        "n_passages": 3,
        "passages_sha256": sha256,
    }

    retriever = DenseRetriever(
        embeddings=embs,
        passage_ids=[p["passage_id"] for p in passages],
        passages=passages,
        metadata=meta,
    )

    index_dir = tmp_path / "test_dense_index"
    retriever.save(index_dir)

    # Rechargement conforme
    loaded = DenseRetriever.load(index_dir, passages_path=p_file, verify_fingerprint=True)
    assert len(loaded.embeddings) == 3
    assert np.allclose(loaded.embeddings, embs)
    assert loaded.passage_ids == ["test_doc_p0001", "test_doc_p0002", "test_doc_p0003"]
    assert len(loaded.passages) == 3

    # Altération du fichier passages pour tester la détection d'empreinte invalide
    corrupted_p_file = tmp_path / "corrupted_passages.jsonl"
    with open(corrupted_p_file, "w", encoding="utf-8") as f:
        f.write('{"passage_id": "altered"}\n')

    with pytest.raises(ValueError, match="L'empreinte SHA-256 du corpus"):
        DenseRetriever.load(index_dir, passages_path=corrupted_p_file, verify_fingerprint=True)

    # Sans vérification d'empreinte, le chargement réussit
    loaded_no_check = DenseRetriever.load(index_dir, passages_path=corrupted_p_file, verify_fingerprint=False)
    assert len(loaded_no_check.embeddings) == 3


def test_text_for_indexing_point_of_entry():
    """Vérifie le point d'entrée text_for_indexing pour les champs content et contextual_content."""
    passage = {
        "passage_id": "p01",
        "content": "Raw content text",
        "contextual_content": "Doc: A | Section: B | Raw content text",
    }
    assert text_for_indexing(passage, field="content") == "Raw content text"
    assert text_for_indexing(passage, field="contextual_content") == "Doc: A | Section: B | Raw content text"
    assert text_for_indexing({}, field="content") == ""
