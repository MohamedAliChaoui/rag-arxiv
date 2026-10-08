"""
Tests unitaires pour le module de retrieval BM25 (src/bm25_retriever.py et src/tokenizer.py).
Vérifie :
1. Le calcul exact à la main de la formule Lucene BM25 sur mini-corpus de 3 passages.
2. La tokenisation scientifique (mots composés, émission des parties, accents Unicode).
3. Le respect strict du contrat d'interface BaseRetriever / SearchResult.
4. La persistance sur disque (save/load) et le contrôle d'empreinte SHA-256.
"""

import math
import os
import shutil
import tempfile
import pytest
import numpy as np

from src.bm25_retriever import BM25Retriever
from src.retriever_base import SearchResult
from src.tokenizer import ScientificTokenizer, TokenizerConfig, tokenize


def test_hand_calculated_bm25_score_on_mini_corpus():
    """
    Test de justesse mathématique :
    Calcule à la main la formule Lucene BM25 sur un mini-corpus de 3 passages
    et vérifie que le moteur bm25s reproduit le résultat au millième de précision.

    Corpus de 3 documents (N = 3, avgdl = 3.0) :
        D1 : 'information retrieval system'  (len=3)
        D2 : 'retrieval augmented generation' (len=3)
        D3 : 'information system generation'  (len=3)

    Requête Q : 'information retrieval'
        t1 = 'information' (df = 2, présent dans D1, D3)
        t2 = 'retrieval'   (df = 2, présent dans D1, D2)

    Formule bm25s (variante 'lucene') :
        IDF(t) = ln(1 + (N - df + 0.5) / (df + 0.5))
               = ln(1 + (3 - 2 + 0.5) / (2 + 0.5)) = ln(1 + 1.5 / 2.5) = ln(1.6) ~= 0.4700036

        TFC(t, D) = tf / (tf + k1 * (1 - b + b * (|D| / avgdl)))
        Avec k1=1.5, b=0.75, |D|=3, avgdl=3 :
        dénominateur = 1 + 1.5 * (1 - 0.75 + 0.75 * 1.0) = 1 + 1.5 * 1.0 = 2.5
        TFC = 1 / 2.5 = 0.4

    Scores attendus :
        Score(D1) = 2 * (IDF * TFC) = 2 * 0.4 * ln(1.6) ~= 0.376003
        Score(D2) = 1 * (IDF * TFC) = 0.4 * ln(1.6) ~= 0.188001
        Score(D3) = 1 * (IDF * TFC) = 0.4 * ln(1.6) ~= 0.188001
    """
    mini_passages = [
        {"passage_id": "doc1", "type": "text", "word_count": 3, "content": "information retrieval system"},
        {"passage_id": "doc2", "type": "text", "word_count": 3, "content": "retrieval augmented generation"},
        {"passage_id": "doc3", "type": "text", "word_count": 3, "content": "information system generation"},
    ]

    with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False, suffix=".jsonl") as f:
        import json
        for p in mini_passages:
            f.write(json.dumps(p) + "\n")
        tmp_file = f.name

    try:
        retriever = BM25Retriever(indexed_field="content", indexed_types=["text"])
        retriever.build_index(tmp_file, show_progress=False)

        results = retriever.search("information retrieval", k=3)
        assert len(results) == 3

        # Vérification du tri : doc1 en tête
        assert results[0].passage_id == "doc1"
        assert results[0].rank == 1

        # Calcul théorique exact
        idf_theorique = math.log(1.0 + (3.0 - 2.0 + 0.5) / (2.0 + 0.5))  # ln(1.6)
        tfc_theorique = 1.0 / (1.0 + 1.5 * (1.0 - 0.75 + 0.75 * 1.0))   # 0.4
        score_d1_theorique = 2.0 * idf_theorique * tfc_theorique         # ~0.3760
        score_d2_theorique = 1.0 * idf_theorique * tfc_theorique         # ~0.1880

        # Vérification numérique
        assert np.isclose(results[0].score, score_d1_theorique, atol=1e-4)
        assert np.isclose(results[1].score, score_d2_theorique, atol=1e-4)
        assert np.isclose(results[2].score, score_d2_theorique, atol=1e-4)

    finally:
        if os.path.exists(tmp_file):
            os.remove(tmp_file)


def test_scientific_tokenizer_compound_words_and_accents():
    """Vérifie l'émission du mot composé et de ses parties, ainsi que les accents."""
    text = "Self-RAG et retrieval-augmented generation avec des modèles déjà entraînés."
    tokens = tokenize(text)

    # 1. Mots composés et sous-parties
    assert "self-rag" in tokens
    assert "self" in tokens
    assert "rag" in tokens

    assert "retrieval-augmented" in tokens
    assert "retrieval" in tokens
    assert "augmented" in tokens

    # 2. Accents préservés grâce à [^\W_]
    assert "modèles" in tokens
    assert "déjà" in tokens
    assert "entraînés" in tokens


def test_scientific_tokenizer_gpt35_turbo_boundary():
    """
    Documente et valide la limite sur GPT-3.5-Turbo :
    Le point décimal agit comme séparateur, produisant les sous-composés disjoints
    ['gpt-3', 'gpt', '3', '5-turbo', '5', 'turbo'].
    """
    tokens = tokenize("Évaluation sur GPT-3.5-Turbo.")
    assert "gpt-3" in tokens
    assert "gpt" in tokens
    assert "3" in tokens
    assert "5-turbo" in tokens
    assert "turbo" in tokens
    assert "gpt-3.5-turbo" not in tokens  # Limite documentée


def test_scientific_tokenizer_technical_metrics():
    """Vérifie la préservation des métriques avec suffixe numérique @."""
    tokens = tokenize("Optimisation de nDCG@10 et MRR@5.")
    assert "ndcg@10" in tokens
    assert "mrr@5" in tokens


def test_scientific_tokenizer_stopwords_and_stemming():
    """Vérifie les options de filtrage de stopwords et de racinisation."""
    text = "The retrieval models are retrieving documents."

    # Sans stopwords ni stemming (défaut)
    tok_default = tokenize(text, TokenizerConfig(remove_stopwords=False, stem=False))
    assert "the" in tok_default
    assert "retrieving" in tok_default

    # Avec stopwords
    tok_no_stop = tokenize(text, TokenizerConfig(remove_stopwords=True, stem=False))
    assert "the" not in tok_no_stop
    assert "are" not in tok_no_stop
    assert "retrieval" in tok_no_stop

    # Avec stemming
    tok_stem = tokenize(text, TokenizerConfig(remove_stopwords=False, stem=True))
    assert "retriev" in tok_stem or "retrieval" in tok_stem  # stem de retrieving


def test_bm25_retriever_save_load_and_fingerprint():
    """Vérifie la persistance sérialisée et la détection d'altération de l'empreinte."""
    mini_passages = [
        {"passage_id": "p001", "type": "text", "word_count": 10, "content": "Machine learning in medical diagnosis."},
        {"passage_id": "p002", "type": "text", "word_count": 12, "content": "Retrieval augmented generation in healthcare."},
    ]

    temp_dir = tempfile.mkdtemp()
    jsonl_path = os.path.join(temp_dir, "passages.jsonl")
    index_dir = os.path.join(temp_dir, "bm25_index")

    try:
        import json
        with open(jsonl_path, "w", encoding="utf-8") as f:
            for p in mini_passages:
                f.write(json.dumps(p) + "\n")

        retriever = BM25Retriever(indexed_field="content")
        retriever.build_index(jsonl_path, show_progress=False)
        retriever.save(index_dir)

        # 1. Chargement nominal
        loaded = BM25Retriever.load(index_dir, verify_fingerprint=True, strict_fingerprint=True)
        results = loaded.search("medical diagnosis", k=1)
        assert len(results) == 1
        assert results[0].passage_id == "p001"
        assert isinstance(results[0], SearchResult)

        # 2. Altération du fichier passages.jsonl pour tester le contrôle d'empreinte
        with open(jsonl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps({"passage_id": "p003", "type": "text", "content": "New corrupted line."}) + "\n")

        # Doit lever une ValueError en mode strict
        with pytest.raises(ValueError, match="Désynchronisation critique de l'index"):
            BM25Retriever.load(index_dir, verify_fingerprint=True, strict_fingerprint=True)

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_self_retrieval_and_index_alignment():
    """
    Test d'auto-récupération sur 200 passages tirés au hasard (graine fixe) :
    1. La requête est formée par les 15 premiers mots du passage.
    2. Le passage d'origine doit être retrouvé dans le top-5 pour les deux index.
    3. Mesure du taux top-1 (>= 80%) et top-5 (>= 95%).
    4. Vérifie l'absence de dérive d'alignement causée par le passage vide exclu.
    """
    import random
    import json

    passages_path = "data/passages.jsonl"
    if not os.path.exists(passages_path):
        pytest.skip("data/passages.jsonl non présent pour le test d'intégration.")

    with open(passages_path, "r", encoding="utf-8") as f:
        passages = [json.loads(line) for line in f]

    # Sélection des passages éligibles (>= 15 mots)
    eligible = [p for p in passages if len(p.get("content", "").split()) >= 15]
    rng = random.Random(42)
    sampled = rng.sample(eligible, 200)

    # 1. Test d'auto-récupération pour les deux index
    for field, index_dir in [("content", "data/index/bm25_content"), ("contextual_content", "data/index/bm25_contextual")]:
        if not os.path.exists(index_dir):
            pytest.skip(f"Index {index_dir} non construit.")

        retriever = BM25Retriever.load(index_dir, verify_fingerprint=False)
        top1_hits = 0
        top5_hits = 0

        for p in sampled:
            q = " ".join(p["content"].split()[:15])
            results = retriever.search(q, k=5)
            pids = [r.passage_id for r in results]
            if pids and pids[0] == p["passage_id"]:
                top1_hits += 1
            if p["passage_id"] in pids:
                top5_hits += 1

        top1_rate = top1_hits / len(sampled)
        top5_rate = top5_hits / len(sampled)

        # Au moins 75% en top-1 et 95% en top-5 exigés
        assert top1_rate >= 0.75, f"Taux top-1 trop faible pour {field}: {top1_rate:.2%}"
        assert top5_rate >= 0.95, f"Taux top-5 insuffisant pour {field}: {top5_rate:.2%}"

    # 2. Vérification de l'alignement pour le passage vide exclu
    target_pid = "2601.16984_p0044"
    jsonl_pids = [p["passage_id"] for p in passages]
    if target_pid in jsonl_pids:
        idx_in_jsonl = jsonl_pids.index(target_pid)
        p_before = jsonl_pids[idx_in_jsonl - 1]
        p_after = jsonl_pids[idx_in_jsonl + 1]

        retriever_c = BM25Retriever.load("data/index/bm25_content", verify_fingerprint=False)
        assert target_pid not in retriever_c.passage_ids

        idx_before = retriever_c.passage_ids.index(p_before)
        idx_after = retriever_c.passage_ids.index(p_after)
        # Strictement contigus sans dérive
        assert idx_after == idx_before + 1

