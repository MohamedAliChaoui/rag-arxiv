"""Tests unitaires pour les fonctions de nettoyage d'indexation (indexing_utils)."""

import pytest
from src.indexing_utils import (
    INDEXING_RULES_VERSION,
    clean_indexing_text,
    text_for_indexing,
)


def test_indexing_rules_version_constant():
    """Vérifie la présence et le format de la constante de version des règles."""
    assert isinstance(INDEXING_RULES_VERSION, str)
    assert len(INDEXING_RULES_VERSION) > 0


def test_clean_indexing_text_color_variants_with_and_without_backslash():
    """Vérifie la suppression des commandes \\color et de la variante sans antislash."""
    # Variante standard avec antislash
    raw_1 = r"order defined by ${\color[rgb]{0.8711,0.5625,0.0195}\pi}$"
    clean_1 = clean_indexing_text(raw_1)
    assert r"\color" not in clean_1
    assert r"\pi" in clean_1

    # Variante simple \color{red}
    raw_2 = r"an alert text \color{red}{warning} message"
    clean_2 = clean_indexing_text(raw_2)
    assert r"\color" not in clean_2
    assert "warning" in clean_2

    # Variante SANS antislash observée dans le passage 2411.07773_p0010
    raw_3 = r"term {\bm{c}}_{{color[rgb]{0.0078,0.6211,0.4492}{\mathcal{D}}}(\pi)}"
    clean_3 = clean_indexing_text(raw_3)
    assert "color[rgb]" not in clean_3
    assert r"\mathcal" not in clean_3
    assert "D" in clean_3  # déballage de \mathcal{D} et retrait de color


def test_clean_indexing_text_scale_and_spacing_modifiers():
    """Vérifie la suppression de \\left, \\right et le remplacement des espacements."""
    raw = r"\left( x + y \right) = a \, b \; c \! d \quad e \qquad f \enspace g"
    cleaned = clean_indexing_text(raw)
    assert r"\left" not in cleaned
    assert r"\right" not in cleaned
    assert r"\," not in cleaned
    assert r"\;" not in cleaned
    assert r"\!" not in cleaned
    assert r"\quad" not in cleaned
    assert r"\qquad" not in cleaned
    assert r"\enspace" not in cleaned
    assert "( x + y ) = a b c d e f g" in cleaned


def test_clean_indexing_text_unwrapping_styles_and_nesting():
    """Vérifie le déballage récursif des commandes de style sans perte de contenu."""
    raw = r"\mathbf{W} \bm{q} \mathrm{loss} \text{PMI} \textbf{bold} \mathit{italic} \mathcal{D}"
    cleaned = clean_indexing_text(raw)
    assert r"\mathbf" not in cleaned
    assert r"\bm" not in cleaned
    assert r"\mathrm" not in cleaned
    assert r"\text" not in cleaned
    assert r"\textbf" not in cleaned
    assert r"\mathit" not in cleaned
    assert r"\mathcal" not in cleaned
    assert "W q loss PMI bold italic D" in cleaned

    # Imbrication récursive (ex: \bm{\mathcal{D}})
    nested = r"\bm{\mathcal{D}}"
    assert clean_indexing_text(nested) == "D"


def test_clean_indexing_text_safety_no_empty_passage():
    """Vérifie qu'un passage contenant du texte ne devient jamais vide."""
    # Si le texte était seulement un symbole
    assert clean_indexing_text(":") == ":"
    assert clean_indexing_text(".") == "."
    # Si tout est nettoyé mais qu'il y avait du texte, le repli protège le passage
    color_only = r"\color[rgb]{1,0,0}"
    res = clean_indexing_text(color_only)
    assert res == color_only  # Fallback sur l'original


def test_text_for_indexing_leaves_passage_dict_unmodified():
    """Vérifie que le dictionnaire source reste strictement intact en mémoire."""
    passage = {
        "passage_id": "test_p01",
        "content": r"Text with ${\bm{q}}$ and \left( x \right)",
        "contextual_content": r"Doc: X | Section: Y | Text with ${\bm{q}}$",
    }
    original_content = passage["content"]
    original_contextual = passage["contextual_content"]

    indexed_content = text_for_indexing(passage, field="content")
    indexed_contextual = text_for_indexing(passage, field="contextual_content")

    # Le texte indexé est nettoyé
    assert r"\bm" not in indexed_content
    assert r"\left" not in indexed_content

    # Les champs originaux du dictionnaire n'ont absolument pas bougé
    assert passage["content"] == original_content
    assert passage["contextual_content"] == original_contextual
