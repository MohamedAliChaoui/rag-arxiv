"""
Script de contrôle qualité du corpus de documents traités (data/processed/).

Vérifications effectuées :
1. Papiers sous le seuil de 1 500 mots (textes anormalement courts ou tronqués).
2. Résidus de balises HTML ou balisage MathML non nettoyé (<math>, <mrow>, etc.).
3. Sections vides (sans blocs de contenu ou avec contenu vide).
4. Doublons (identifiants arXiv identiques ou titres dupliqués).
5. Papiers avec un nombre anormalement élevé de tableaux (outliers statistiques).
"""

import json
import os
import re
import statistics
from collections import Counter
from typing import Any, Dict, List


def check_corpus(processed_dir: str = "data/processed") -> Dict[str, Any]:
    if not os.path.exists(processed_dir):
        print(f"[!] Dossier {processed_dir} introuvable.")
        return {}

    files = [f for f in os.listdir(processed_dir) if f.endswith(".json")]
    total_files = len(files)

    short_papers = []
    html_residuals = []
    empty_sections = []
    seen_ids = set()
    id_duplicates = []
    titles_map: Dict[str, List[str]] = {}
    table_counts = []
    paper_table_map = []

    # Regex pour détecter les balises HTML/XML résiduelles (ex: <div ...>, <math...>, <mrow>)
    tag_pattern = re.compile(r"<\s*([a-zA-Z0-9_\-]+)(?:\s+[^>]*)?>|<!--.*?-->", re.DOTALL)

    for filename in files:
        filepath = os.path.join(processed_dir, filename)
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        arxiv_id = data.get("arxiv_id", filename)
        title = data.get("title", "Sans titre").strip()
        norm_title = re.sub(r"\s+", " ", re.sub(r"[^\w\s]", "", title.lower())).strip()

        # 1. Doublons ID
        if arxiv_id in seen_ids:
            id_duplicates.append(arxiv_id)
        seen_ids.add(arxiv_id)

        # Doublons Titre
        titles_map.setdefault(norm_title, []).append(arxiv_id)

        # 2. Papiers sous 1500 mots
        total_words = data.get("total_words", 0)
        if total_words < 1500:
            short_papers.append({
                "arxiv_id": arxiv_id,
                "title": title,
                "words": total_words,
            })

        # 3. Tables count
        num_tables = data.get("block_counts", {}).get("table", 0)
        table_counts.append(num_tables)
        paper_table_map.append({"arxiv_id": arxiv_id, "title": title, "tables": num_tables})

        # 4. Parcours des sections et blocs
        sections = data.get("sections", [])
        for sec in sections:
            sec_id = sec.get("section_id", "inconnu")
            sec_heading = sec.get("heading", "Sans titre")
            blocks = sec.get("blocks", [])

            if not blocks:
                empty_sections.append({
                    "arxiv_id": arxiv_id,
                    "section_id": sec_id,
                    "heading": sec_heading,
                    "reason": "Section sans aucun bloc de contenu",
                })
            else:
                for b_idx, block in enumerate(blocks):
                    content = block.get("content", "").strip()
                    if not content:
                        empty_sections.append({
                            "arxiv_id": arxiv_id,
                            "section_id": sec_id,
                            "heading": sec_heading,
                            "reason": f"Bloc vide à l'indice {b_idx}",
                        })
                    else:
                        # Vérification résidus de balises HTML/MathML
                        # Tolérance pour les symboles mathématiques < et > sans forme de balise HTML
                        matches = tag_pattern.findall(content)
                        # Ignorer les faux positifs d'inégalités ou notations markdown
                        real_tags = [
                            m for m in matches
                            if m.lower() in [
                                "math", "mrow", "mi", "mo", "mn", "table", "tr", "td", "th",
                                "div", "span", "p", "a", "img", "svg", "section", "body"
                            ]
                        ]
                        if real_tags:
                            html_residuals.append({
                                "arxiv_id": arxiv_id,
                                "section_id": sec_id,
                                "tag_sample": list(set(real_tags)),
                                "snippet": content[:120],
                            })

    # Détection des doublons de titres
    title_duplicates = [
        {"title": t, "arxiv_ids": ids} for t, ids in titles_map.items() if len(ids) > 1
    ]

    # Détection des outliers sur le nombre de tableaux (seuil statistique IQR : Q3 + 1.5 * IQR)
    outlier_tables = []
    if table_counts:
        table_counts_sorted = sorted(table_counts)
        q1 = statistics.quantiles(table_counts_sorted, n=4)[0]
        q3 = statistics.quantiles(table_counts_sorted, n=4)[2]
        iqr = q3 - q1
        outlier_threshold = max(20, int(q3 + 2.5 * iqr))  # Minimum 20 pour éviter les faux positifs

        for pt in paper_table_map:
            if pt["tables"] >= outlier_threshold:
                outlier_tables.append(pt)

    # Affichage du rapport
    print("\n" + "=" * 75)
    print(f"       RAPPORT DE CONTRÔLE QUALITÉ DU CORPUS ({total_files} ARTICLES)")
    print("=" * 75)

    # 1. Doublons
    print(f" [1] DOUBLONS :")
    print(f"     - Identifiants arXiv dupliqués : {len(id_duplicates)}")
    print(f"     - Articles avec titre identique : {len(title_duplicates)}")
    if title_duplicates:
        for td in title_duplicates:
            print(f"       * Titre : {td['title'][:60]}... -> IDs: {td['arxiv_ids']}")

    # 2. Papiers courts (< 1500 mots)
    print(f"\n [2] ARTICLES COURTS (< 1 500 mots) : {len(short_papers)}")
    if short_papers:
        for sp in short_papers:
            print(f"     - [{sp['arxiv_id']}] {sp['words']} mots : {sp['title'][:65]}...")
    else:
        print("     [OK] Aucun article sous 1 500 mots.")

    # 3. Résidus HTML / MathML
    print(f"\n [3] RÉSIDUS DE BALISES HTML / MathML : {len(html_residuals)}")
    if html_residuals:
        for hr in html_residuals[:5]:
            print(f"     - [{hr['arxiv_id']}] sec={hr['section_id']} tags={hr['tag_sample']} : {hr['snippet']}...")
        if len(html_residuals) > 5:
            print(f"       ... et {len(html_residuals) - 5} autres résidus.")
    else:
        print("     [OK] Aucun résidu de balises HTML/MathML détecté dans les blocs textuels.")

    # 4. Sections vides
    print(f"\n [4] SECTIONS OU BLOCS VIDES : {len(empty_sections)}")
    if empty_sections:
        for es in empty_sections[:5]:
            print(f"     - [{es['arxiv_id']}] {es['section_id']} ({es['heading']}) : {es['reason']}")
        if len(empty_sections) > 5:
            print(f"       ... et {len(empty_sections) - 5} autres sections vides.")
    else:
        print("     [OK] Aucune section vide détectée.")

    # 5. Outliers tableaux
    print(f"\n [5] NOMBRE ANORMAL DE TABLEAUX (seuil >= {outlier_threshold} tables) : {len(outlier_tables)}")
    if outlier_tables:
        for ot in sorted(outlier_tables, key=lambda x: x['tables'], reverse=True):
            print(f"     - [{ot['arxiv_id']}] {ot['tables']} tableaux : {ot['title'][:60]}...")
    else:
        print(f"     [OK] Aucun outlier extrême sur les tableaux (max observé: {max(table_counts) if table_counts else 0}).")

    print("=" * 75 + "\n")

    return {
        "total_files": total_files,
        "id_duplicates": id_duplicates,
        "title_duplicates": title_duplicates,
        "short_papers": short_papers,
        "html_residuals": html_residuals,
        "empty_sections": empty_sections,
        "outlier_tables": outlier_tables,
    }


if __name__ == "__main__":
    check_corpus()
