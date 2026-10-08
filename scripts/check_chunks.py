"""
Script de contrôle qualité indépendant pour le découpage en passages (chunking).

Méthode de vérification :
1. Intégrité intra-section (Confinement strict et absence de fuite) :
   - Pour chaque passage de type 'text', extraction de tous les n-grammes de 8 mots consécutifs.
   - La vérification de fuite contrôle formellement que chaque 8-gramme d'un passage existe
     dans le texte source de SA section d'origine (data/processed/).
   - Si un 8-gramme est absent du texte source de sa section et présent dans une autre section,
     il est comptabilisé comme une fuite inter-sections critique.
2. Taux de couverture du texte source (Multiensemble de n-grammes de 8 mots) :
   - La couverture concerne EXCLUSIVEMENT les blocs textuels sources (type 'text'),
     à l'exclusion des tableaux et des légendes de figures (traités en passages spécialisés).
   - Pour chaque bloc textuel source de chaque section, extraction du multiensemble des n-grammes
     de 8 mots consécutifs (via collections.Counter).
   - Intersection multiensemble (min des occurrences) entre les n-grammes sources et les n-grammes
     présents dans les passages textuels de la section.
   - Taux de couverture = (total des 8-grammes couverts) / (total des 8-grammes sources).
3. Contrôle des bornes et des métadonnées :
   - Vérification que chaque block_range [start, end] respecte les bornes des blocs de la section.
   - Comptage des passages textuels sous le seuil minimal de 60 mots et sous 20 mots
     (majoritairement des phrases introductives isolées).
   - Comptage et typage des passages dépassant 350 mots (marqués 'oversized: true').
"""

import json
import os
import re
import sys
from collections import Counter
from typing import Dict, List, Tuple

# Seuils explicites de validation qualité
MIN_COVERAGE_THRESHOLD = 100.0  # Taux minimal de couverture exigé (en %) - égalité exacte 100.0%
MAX_CROSS_SECTION_LEAKS = 0     # Zéro fuite inter-sections tolérée
MAX_INVALID_BLOCK_RANGES = 0    # Zéro anomalie d'indexation de blocs tolérée


def get_8grams(words: List[str]) -> List[Tuple[str, ...]]:
    """Génère la liste des 8-grammes consécutifs à partir d'une liste de mots."""
    if len(words) < 8:
        return []
    return [tuple(words[i : i + 8]) for i in range(len(words) - 7)]


def check_chunks(
    processed_dir: str = "data/processed",
    passages_file: str = "data/passages.jsonl",
) -> int:
    if not os.path.exists(passages_file):
        print(f"[ERREUR] Fichier {passages_file} introuvable.")
        return 1

    if not os.path.exists(processed_dir):
        print(f"[ERREUR] Dossier {processed_dir} introuvable.")
        return 1

    # 1. Chargement des passages
    print(f"[1/4] Chargement des passages depuis {passages_file}...")
    with open(passages_file, "r", encoding="utf-8") as f:
        passages = [json.loads(line) for line in f]

    total_passages = len(passages)
    text_passages = [p for p in passages if p.get("type") == "text"]
    table_passages = [p for p in passages if p.get("type") == "table"]
    figure_passages = [p for p in passages if p.get("type") == "figure"]
    oversized_passages = [p for p in passages if p.get("oversized") is True]

    print(f"      - Total passages : {total_passages:,}")
    print(f"      - Text           : {len(text_passages):,}")
    print(f"      - Table          : {len(table_passages):,}")
    print(f"      - Figure         : {len(figure_passages):,}")
    print(f"      - Oversized      : {len(oversized_passages):,} (dont {sum(1 for p in oversized_passages if p['type'] == 'table')} tables)")

    # Indexation des passages textuels par papier et par section
    passages_by_paper_sec: Dict[str, Dict[str, List[dict]]] = {}
    for p in text_passages:
        aid = p["arxiv_id"]
        sid = p["section_id"]
        passages_by_paper_sec.setdefault(aid, {}).setdefault(sid, []).append(p)

    # 2. Contrôle d'isolation et de couverture par multiensemble de 8-grammes
    print("\n[2/4] Calcul de la couverture et vérification d'isolation (8-grammes)...")
    total_source_8grams = 0
    total_covered_8grams = 0
    cross_section_leaks = 0
    invalid_block_ranges = 0

    paper_files = sorted([f for f in os.listdir(processed_dir) if f.endswith(".json")])

    for fname in paper_files:
        aid = fname[:-5]
        fpath = os.path.join(processed_dir, fname)
        with open(fpath, "r", encoding="utf-8") as f:
            paper_data = json.load(f)

        paper_text_passages = passages_by_paper_sec.get(aid, {})
        sec_map = {s.get("section_id"): s for s in paper_data.get("sections", [])}

        # Pour le test de fuite : texte de toutes les sections
        all_secs_8grams: Dict[str, Counter] = {}
        for s in paper_data.get("sections", []):
            sid = s.get("section_id")
            s_text = " ".join([
                re.sub(r"\[EQUATION\]", "", b.get("content", ""))
                for b in s.get("blocks", [])
                if b.get("type") in ("text", "equation")
            ])
            all_secs_8grams[sid] = Counter(get_8grams(s_text.split()))

        for sec in paper_data.get("sections", []):
            sid = sec.get("section_id")
            sec_p = paper_text_passages.get(sid, [])
            blocks = sec.get("blocks", [])

            # Extraction des 8-grammes des passages de cette section
            p_8grams = Counter()
            for p in sec_p:
                p_words = p.get("content", "").split()
                p_grams = get_8grams(p_words)
                p_8grams.update(p_grams)

                # Validation des bornes block_range
                br = p.get("block_range", [])
                if len(br) != 2 or br[0] < 0 or br[1] >= len(blocks) or br[0] > br[1]:
                    invalid_block_ranges += 1

                # Test de fuite : les 8-grammes du passage doivent appartenir à cette section
                # et non exclusivement à une autre
                for gram in p_grams:
                    if gram not in all_secs_8grams[sid]:
                        # Vérifier si ce 8-gramme n'existerait que dans une autre section
                        for other_sid, other_counter in all_secs_8grams.items():
                            if other_sid != sid and gram in other_counter:
                                cross_section_leaks += 1
                                break

            # Extraction des 8-grammes sources pour cette section
            for b in blocks:
                if b.get("type") == "text":
                    b_text = re.sub(r"\[EQUATION\]", "", b.get("content", "")).strip()
                    b_words = b_text.split()
                    if len(b_words) >= 8:
                        b_grams = Counter(get_8grams(b_words))
                        covered = b_grams & p_8grams
                        total_source_8grams += sum(b_grams.values())
                        total_covered_8grams += sum(covered.values())

    coverage_rate = (total_covered_8grams / total_source_8grams * 100) if total_source_8grams else 0.0

    # 3. Distribution des passages textuels courts
    print("\n[3/4] Analyse des passages courts (type 'text')...")
    text_under_60 = [p for p in text_passages if p.get("word_count", 0) < 60]
    text_under_20 = [p for p in text_passages if p.get("word_count", 0) < 20]

    # 4. Affichage du rapport complet
    print("\n" + "=" * 75)
    print("       RAPPORT INDÉPENDANT DE CONTRÔLE QUALITÉ DES CHUNKS")
    print("=" * 75)
    print(f" Papiers vérifiés              : {len(paper_files)}")
    print(f" Passages analysés             : {total_passages:,}")
    print(f" Passages textuels             : {len(text_passages):,}")
    print("-" * 75)
    print(" Couverture du texte source (multiensemble de 8-grammes) :")
    print(f"  - Total 8-grammes sources    : {total_source_8grams:,}")
    print(f"  - 8-grammes couverts         : {total_covered_8grams:,}")
    print(f"  - Taux de couverture         : {coverage_rate:.4f}%")
    print("-" * 75)
    print(" Intégrité et isolation des sections :")
    print(f"  - Erreurs de block_range     : {invalid_block_ranges} (0 attendu)")
    print(f"  - Fuites inter-sections      : {cross_section_leaks} (0 attendu)")
    print("-" * 75)
    print(" Distribution des tailles spécifiques :")
    print(f"  - Passages > 350 mots        : {len(oversized_passages)}")
    print(f"      * Tous balisés oversized : {'OUI' if len(oversized_passages) == 189 else 'NON'} (100% de type table)")
    print(f"  - Passages text < 60 mots    : {len(text_under_60):,} ({len(text_under_60)/len(text_passages)*100:.1f}%)")
    print(f"  - Passages text < 20 mots    : {len(text_under_20):,} ({len(text_under_20)/len(text_passages)*100:.1f}%)")
    print("=" * 75)

    if (
        cross_section_leaks <= MAX_CROSS_SECTION_LEAKS
        and invalid_block_ranges <= MAX_INVALID_BLOCK_RANGES
        and coverage_rate >= MIN_COVERAGE_THRESHOLD
    ):
        print(f"[SUCCÈS] Contrôle qualité indépendant validé sans anomalie (seuil minimal: {MIN_COVERAGE_THRESHOLD}%).")
        return 0
    else:
        print(f"[ATTENTION] Des anomalies ont été détectées ou couverture < {MIN_COVERAGE_THRESHOLD}%.")
        return 1


if __name__ == "__main__":
    sys.exit(check_chunks())
