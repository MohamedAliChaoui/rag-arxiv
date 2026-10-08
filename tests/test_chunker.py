"""
Tests unitaires pour le module de découpage en passages (src/chunker.py).
Vérifie la robustesse scientifique, l'isolation des sections, l'overlap configurable,
la traçabilité block_range, la gestion des tableaux et des chemins hiérarchiques.
"""

import pytest
from src.chunker import (
    split_sentences,
    build_section_path,
    chunk_markdown_table,
    _chunk_text_units,
    chunk_section,
    chunk_paper,
)


def test_split_sentences_scientific():
    """Vérifie la protection des abréviations scientifiques, références et nombres décimaux."""
    text = (
        "Smith et al. evaluated the model on RAG benchmarks. "
        "As seen in Fig. 1 and Tab. 2, the score reached 0.95 vs. 0.82 previously. "
        "Is this consistent? Yes, i.e., across all tests."
    )
    sents = split_sentences(text)
    assert len(sents) == 4
    assert "et al." in sents[0]
    assert "Fig. 1" in sents[1]
    assert "Tab. 2" in sents[1]
    assert "0.95" in sents[1]
    assert "vs." in sents[1]
    assert sents[2] == "Is this consistent?"
    assert "Yes, i.e." in sents[3]


def test_split_sentences_preserves_casing_and_urls():
    """
    Test de non-régression : vérifie que la protection des abréviations
    restitue rigoureusement la casse originale et n'altère pas Quoref., No./no.,
    ni les URL contenant 'sec.' (ex: https://www.sec.gov).
    """
    text = (
        "We evaluate on Quoref. Results are shown in Table 1. "
        "Grant No. 12345 was received, but no. 67890 was rejected. "
        "More details are available at https://www.sec.gov/files/cf-frm.pdf. Next study follows."
    )
    sents = split_sentences(text)
    assert len(sents) == 5
    # 1. Quoref ne doit pas être altéré en QuoRef et doit être scindé en fin de phrase
    assert sents[0] == "We evaluate on Quoref."
    assert sents[1] == "Results are shown in Table 1."

    # 2. No. et no. conservent strictement leur casse respective
    assert "No. 12345" in sents[2]
    assert "no. 67890" in sents[2]

    # 3. L'URL contenant 'sec.' conserve son écriture minuscule exacte (pas Sec.gov)
    assert "https://www.sec.gov/files/cf-frm.pdf" in sents[3]
    assert sents[4] == "Next study follows."


def test_build_section_path_hierarchy_and_appendices():
    """Vérifie la résolution hiérarchique, les annexes (A...) et le fallback des parents absents."""
    smap = {
        "abstract": "Abstract",
        "S1": "1 Introduction",
        "S1.SS1": "1.1 Motivation",
        "A1": "Appendix A Technical Proofs",
        "A1.SS1": "A.1 Soundness Lemma",
    }

    # Abstract
    p, missing = build_section_path(smap, {"section_id": "abstract", "heading": "Abstract"})
    assert p == "Abstract"
    assert missing is False

    # Section standard
    p, missing = build_section_path(smap, {"section_id": "S1.SS1", "heading": "1.1 Motivation"})
    assert p == "1 Introduction > 1.1 Motivation"
    assert missing is False

    # Annexe
    p, missing = build_section_path(smap, {"section_id": "A1.SS1", "heading": "A.1 Soundness Lemma"})
    assert p == "Appendix A Technical Proofs > A.1 Soundness Lemma"
    assert missing is False

    # Parent absent (ex: S2.SS3 où S2 n'existe pas dans le map)
    p, missing = build_section_path(smap, {"section_id": "S2.SS3", "heading": "2.3 Sub Analysis"})
    assert p == "2.3 Sub Analysis"
    assert missing is True


def test_chunk_markdown_table_splitting():
    """Vérifie le découpage de tableaux volumineux avec répétition de la légende et des en-têtes."""
    caption = "[Table 1: Benchmark Results]"
    header = "| Model | Precision | Recall | F1 | Latency |"
    sep = "| --- | --- | --- | --- | --- |"
    rows = [f"| Model_{i} | 0.8{i} | 0.7{i} | 0.7{i} | 1{i}ms |" for i in range(100)]
    table_text = "\n".join([caption, header, sep] + rows)

    chunks = chunk_markdown_table(table_text, max_words=100)
    assert len(chunks) > 1

    for c in chunks:
        assert caption in c
        assert header in c
        assert sep in c
        # Chaque chunk doit avoir au moins une ligne de données
        assert "| Model_" in c


def test_section_isolation_never_crosses():
    """Garantit qu'aucun passage ne combine des blocs de deux sections différentes."""
    paper = {
        "arxiv_id": "2401.99999v1",
        "clean_id": "2401.99999",
        "title": "Strict Isolation Study",
        "month_stratum": "2024-01",
        "sections": [
            {
                "section_id": "S1",
                "heading": "1 Introduction",
                "blocks": [
                    {"type": "text", "content": "Paragraph in Introduction section. " * 15}
                ],
            },
            {
                "section_id": "S2",
                "heading": "2 Methods",
                "blocks": [
                    {"type": "text", "content": "Paragraph in Methods section. " * 15}
                ],
            },
        ],
    }

    passages, _ = chunk_paper(paper, target_words=100, max_words=150)
    assert len(passages) >= 2

    for p in passages:
        if p["section_id"] == "S1":
            assert "Introduction" in p["section_path"]
            assert "Methods" not in p["content"]
        elif p["section_id"] == "S2":
            assert "Methods" in p["section_path"]
            assert "Introduction" not in p["content"]


def test_overlap_configurable_zero_vs_forty():
    """Vérifie le comportement de l'overlap : 0 mot vs 40 mots."""
    units = [
        {"block_idx": 0, "text": f"Sentence number {i} with several words for length testing.", "words": 10}
        for i in range(40)
    ]

    # Test avec overlap = 0
    chunks_0 = _chunk_text_units(units, target_words=50, max_words=70, overlap_words=0)
    # Vérification qu'aucune phrase n'est partagée entre chunk 0 et chunk 1
    texts_0_c0 = set(chunks_0[0]["content"].split(". "))
    texts_0_c1 = set(chunks_0[1]["content"].split(". "))
    assert len(texts_0_c0.intersection(texts_0_c1)) == 0

    # Test avec overlap = 40
    chunks_40 = _chunk_text_units(units, target_words=50, max_words=70, overlap_words=40)
    # Vérification qu'une phrase au moins est partagée
    words_c0 = chunks_40[0]["content"].split()
    words_c1 = chunks_40[1]["content"].split()
    overlap_count = len(set(words_c0).intersection(set(words_c1)))
    assert overlap_count > 0


def test_intra_section_short_text_merging():
    """
    Vérifie la fusion des paragraphes consécutifs jusqu'à ~250 mots,
    et la fusion intra-section avec le voisin pour les blocs < 60 mots.
    """
    sec = {
        "section_id": "S3",
        "heading": "3 Experiments",
        "blocks": [
            {"type": "text", "content": "Short intro to the experiment. " * 3},  # ~15 mots
            {"type": "table", "content": "| A | B |\n| --- | --- |\n| 1 | 2 |"},
            {"type": "text", "content": "Discussion of the experiment results. " * 15},  # ~75 mots
        ],
    }

    passages, _ = chunk_section(
        sec=sec,
        section_path="3 Experiments",
        title="Test Paper",
        arxiv_id="2401.00001",
        clean_id="2401.00001",
        month_stratum="2024-01",
        passage_counter_start=1,
        min_words=60,
        max_words=350,
    )

    # Le bloc de 15 mots doit avoir fusionné avec le bloc de 75 mots
    # Le tableau doit rester un passage distinct
    types = [p["type"] for p in passages]
    assert types == ["table", "text"] or types == ["text", "table"]

    text_p = next(p for p in passages if p["type"] == "text")
    assert text_p["word_count"] >= 60
    assert text_p["block_range"] == [0, 2]  # Couvre le bloc 0 et le bloc 2
    assert "Short intro" in text_p["content"]
    assert "Discussion of the experiment" in text_p["content"]


def test_block_range_and_char_count_presence():
    """Vérifie la présence et validité des champs block_range, char_count et contextual_content."""
    paper = {
        "arxiv_id": "2401.12345v1",
        "clean_id": "2401.12345",
        "title": "Document Breadcrumb Test",
        "month_stratum": "2024-01",
        "sections": [
            {
                "section_id": "S1",
                "heading": "1 Overview",
                "blocks": [
                    {"type": "text", "content": "First block in section overview. " * 10},
                    {"type": "figure", "content": "[Figure 1: Architectural diagram of the pipeline]"},
                ],
            }
        ],
    }

    passages, _ = chunk_paper(paper)
    assert len(passages) == 2

    for p in passages:
        assert "block_range" in p
        assert isinstance(p["block_range"], list)
        assert len(p["block_range"]) == 2
        assert p["block_range"][0] <= p["block_range"][1]

        assert "char_count" in p
        assert p["char_count"] == len(p["content"])

        assert "contextual_content" in p
        assert "Document: Document Breadcrumb Test" in p["contextual_content"]
        assert "Section: 1 Overview" in p["contextual_content"]
        assert p["content"] in p["contextual_content"]


def test_oversized_flag_for_long_passages():
    """Vérifie que le champ oversized: True est bien apposé aux passages > 350 mots."""
    # Tableau indivisible dont une seule ligne dépasse 350 mots
    long_row = "| " + " ".join(["value"] * 400) + " |"
    table_content = "[Table 1: Huge table]\n| Col1 |\n| --- |\n" + long_row
    paper = {
        "arxiv_id": "2401.99999v1",
        "clean_id": "2401.99999",
        "title": "Oversized Test Paper",
        "sections": [
            {
                "section_id": "S1",
                "heading": "1 Results",
                "blocks": [
                    {"type": "table", "content": table_content},
                    {"type": "text", "content": "Short normal text passage with enough words to stand alone. " * 5},
                ],
            }
        ],
    }

    passages, _ = chunk_paper(paper)
    table_p = next(p for p in passages if p["type"] == "table")
    text_p = next(p for p in passages if p["type"] == "text")

    assert table_p["word_count"] > 350
    assert table_p.get("oversized") is True

    assert text_p["word_count"] <= 350
    assert "oversized" not in text_p

