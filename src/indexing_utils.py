"""Utilitaires de normalisation et nettoyage de texte pour l'indexation (BM25 et Dense).

Ces règles de nettoyage sont appliquées UNIQUEMENT lors de la projection textuelle
pour l'indexation. Le champ source 'content' dans passages.jsonl demeure strictement
intact (pour l'affichage, les citations et l'audit).
"""

from __future__ import annotations

import re
from typing import Any, Dict

# Version des règles d'indexation
# Permet d'invalider les index BM25, denses et checkpoints si les règles changent
INDEXING_RULES_VERSION = "1.0"

# Motifs de nettoyage compilés
_RE_COLOR = re.compile(r'\\?color(?:\[[^\]]*\])?\{[^\}]*\}')
_RE_LEFT_RIGHT = re.compile(r'\\(?:left|right)\b')
_RE_SPACING = re.compile(r'\\[,;!]|\\(?:quad|qquad|enspace)\b')
_STYLE_CMDS = r'(?:mathbf|bm|mathrm|text|textbf|mathit|mathcal|mathsf|boldsymbol)'
_RE_STYLES = re.compile(rf'\\{_STYLE_CMDS}\{{([^{{}}]*)\}}')
_RE_MULTI_SPACES = re.compile(r'[ \t]+')


def clean_indexing_text(text: str) -> str:
    """Nettoie le texte en retirant le bruit de mise en forme sans contenu sémantique.

    Règles conservatrices :
    1. Suppression des balises de couleur (avec ou sans antislash, ex: \\color[rgb]{...} ou {color[rgb]{...})
    2. Suppression des modificateurs de taille de délimiteurs (\\left, \\right)
    3. Remplacement des espacements mathématiques par un espace standard (\\,, \\;, \\!, \\quad, etc.)
    4. Déballage récursif des commandes de style/fonte tout en conservant leur contenu (\\mathbf{x} -> x, \\bm{q} -> q)
    5. Normalisation des espaces consécutifs
    """
    if not text:
        return ""

    t = text

    # 1. Suppression des balises de couleur (avec ou sans antislash)
    t = _RE_COLOR.sub('', t)

    # 2. Suppression de \left et \right
    t = _RE_LEFT_RIGHT.sub('', t)

    # 3. Remplacement des espacements mathématiques par un espace
    t = _RE_SPACING.sub(' ', t)

    # 4. Déballage récursif des commandes de style (jusqu'à 3 passes pour les imbrications)
    for _ in range(3):
        t = _RE_STYLES.sub(r'\1', t)

    # 5. Normalisation des espaces horizontaux multiples
    t = _RE_MULTI_SPACES.sub(' ', t)

    # Si le nettoyage a vidé un texte qui contenait des caractères non-blancs, on préserve l'original
    if not t.strip() and text.strip():
        return text

    return t


def text_for_indexing(passage: Dict[str, Any], field: str = "content") -> str:
    """Point d'entrée unique de projection textuelle pour l'indexation (BM25 et Dense).

    Applique la fonction clean_indexing_text au champ demandé.
    Garantit que le dictionnaire original 'passage' n'est jamais modifié.
    """
    raw_text = passage.get(field, "") or ""
    return clean_indexing_text(raw_text)
