"""
Module de parsing et nettoyage sémantique du HTML expérimental d'arXiv (LaTeXML).

Règles de parsing appliquées :
1. Suppression intégrale des sections de références / bibliographie.
2. Traitement des formules mathématiques :
   - Inline : extraction de l'attribut LaTeX alttext et délimitation par '$...$'.
   - Hors-bloc : délimitation '$$...$$', et remplacement par '[EQUATION]' si > 200 caractères.
3. Traitement des figures : suppression des images binaires, conservation de la légende '[Figure X: ...]'.
4. Traitement des tableaux : conservation de la légende et conversion tabulaire en format Markdown.
5. Découpage par section : chaque bloc contient son type ('text', 'table', 'equation', 'figure')
   et son section_id d'origine pour permettre la citation ultérieure 'papier + section'.
"""

import re
from typing import Any, Dict, List
from bs4 import BeautifulSoup


def _clean_whitespace(text: str) -> str:
    """Réduit les suites d'espaces et sauts de ligne à un espace simple."""
    if not text:
        return ""
    return re.sub(r"\s+", " ", text).strip()


def _convert_table_to_markdown(figure_or_table) -> str:
    """
    Convertit un tableau HTML LaTeXML en représentation Markdown textuelle
    précédée de sa légende éventuelle.
    """
    caption_el = figure_or_table.find("figcaption")
    caption_text = (
        _clean_whitespace(caption_el.get_text(" ", strip=True))
        if caption_el
        else ""
    )

    table_el = (
        figure_or_table
        if figure_or_table.name == "table"
        else figure_or_table.find("table")
    )
    if not table_el:
        return f"[{caption_text}]" if caption_text else ""

    rows = table_el.find_all("tr")
    grid: List[List[str]] = []
    max_cols = 0

    for tr in rows:
        cells = tr.find_all(["th", "td"])
        row_cells = [_clean_whitespace(c.get_text(" ", strip=True)) for c in cells]
        if row_cells:
            grid.append(row_cells)
            max_cols = max(max_cols, len(row_cells))

    if not grid or max_cols == 0:
        return f"[{caption_text}]" if caption_text else ""

    # Normalisation du nombre de colonnes pour toutes les lignes
    for row in grid:
        while len(row) < max_cols:
            row.append("")

    lines = []
    if caption_text:
        lines.append(f"[{caption_text}]")

    # En-têtes (première ligne)
    header = grid[0]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("| " + " | ".join(["---"] * max_cols) + " |")

    # Lignes de données
    for row in grid[1:]:
        lines.append("| " + " | ".join(row) + " |")

    return "\n".join(lines)


def parse_paper_html(html_content: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
    """
    Parse le document HTML d'un article arXiv et produit une structure JSON
    enrichie de blocs typés par section, sans bibliographie.
    """
    soup = BeautifulSoup(html_content, "html.parser")

    # 1. Élimination complète de la bibliographie / références
    for bib in soup.find_all(class_="ltx_bibliography"):
        bib.decompose()

    for s in soup.find_all(["section", "div"]):
        t = s.find(["h1", "h2", "h3", "h4", "h5", "h6"])
        if t and any(w in t.get_text().lower() for w in ["references", "bibliography"]):
            s.decompose()

    # 2. Nettoyage des formules inline : conversion MathML -> LaTeX $...$
    for math in soup.find_all("math"):
        # Les équations hors-bloc sont traitées à part comme blocs autonomes
        if math.find_parent(class_=["ltx_equation", "ltx_equationgroup"]):
            continue
        alt = math.get("alttext")
        if alt:
            math.replace_with(f" ${alt.strip()}$ ")
        else:
            raw_text = math.get_text(" ", strip=True)
            math.replace_with(f" {raw_text} " if raw_text else " ")

    sections_output: List[Dict[str, Any]] = []

    # 3. Traitement de l'Abstract si présent dans une balise dédiée
    abstract_el = soup.find(class_="ltx_abstract")
    if abstract_el:
        heading_el = abstract_el.find(["h1", "h2", "h3", "h4", "h5", "h6"])
        heading = (
            _clean_whitespace(heading_el.get_text(" ", strip=True))
            if heading_el
            else "Abstract"
        )
        abstract_blocks = []
        for p in abstract_el.find_all("p"):
            p_text = _clean_whitespace(p.get_text(" ", strip=True))
            if p_text:
                abstract_blocks.append({
                    "section_id": "abstract",
                    "type": "text",
                    "content": p_text,
                })
        if abstract_blocks:
            sections_output.append({
                "section_id": "abstract",
                "heading": heading,
                "blocks": abstract_blocks,
            })
        abstract_el.decompose()

    # 4. Parcours des sections du document
    section_tags = soup.find_all("section")

    # Cas de secours si aucune balise <section> n'existe (structure plate)
    if not section_tags:
        main_body = soup.find("body") or soup
        section_tags = [main_body]

    for sec in section_tags:
        sec_id = sec.get("id", "") or f"sec_{len(sections_output) + 1}"
        title_el = sec.find(["h1", "h2", "h3", "h4", "h5", "h6"])
        heading = (
            _clean_whitespace(title_el.get_text(" ", strip=True))
            if title_el
            else ""
        )

        blocks: List[Dict[str, Any]] = []
        processed_elements = set()

        for el in sec.find_all(["figure", "table", "div", "p"]):
            if el in processed_elements:
                continue

            # Évite d'attribuer un bloc d'une sous-section à la section parente
            if el.find_parent("section") != sec:
                continue

            classes = el.get("class", [])

            # --- Figures ---
            if el.name == "figure" and "ltx_figure" in classes:
                caption = el.find("figcaption")
                caption_text = (
                    _clean_whitespace(caption.get_text(" ", strip=True))
                    if caption
                    else ""
                )
                if caption_text:
                    blocks.append({
                        "section_id": sec_id,
                        "type": "figure",
                        "content": f"[{caption_text}]",
                    })
                for child in el.descendants:
                    processed_elements.add(child)
                processed_elements.add(el)

            # --- Tableaux ---
            elif (el.name == "figure" and "ltx_table" in classes) or (
                el.name == "table" and "ltx_tabular" in classes
            ):
                table_md = _convert_table_to_markdown(el)
                if table_md:
                    blocks.append({
                        "section_id": sec_id,
                        "type": "table",
                        "content": table_md,
                    })
                for child in el.descendants:
                    processed_elements.add(child)
                processed_elements.add(el)

            # --- Équations hors-bloc ---
            elif any(
                c in classes for c in ["ltx_equation", "ltx_equationgroup"]
            ):
                math_tag = el.find("math")
                eq_text = (
                    math_tag.get("alttext", "")
                    if math_tag
                    else el.get_text(" ", strip=True)
                )
                eq_text = _clean_whitespace(eq_text)

                # Règle demandée : remplacement par [EQUATION] si > 200 caractères
                if len(eq_text) > 200:
                    eq_content = "[EQUATION]"
                else:
                    eq_content = f"$${eq_text}$$"

                blocks.append({
                    "section_id": sec_id,
                    "type": "equation",
                    "content": eq_content,
                })
                for child in el.descendants:
                    processed_elements.add(child)
                processed_elements.add(el)

            # --- Paragraphes de texte standard ---
            elif el.name == "p":
                # Évite d'extraire le texte d'un conteneur déjà traité (légende, titre, etc.)
                if el.find_parent(
                    ["figure", "figcaption", "h1", "h2", "h3", "h4", "h5", "h6"]
                ):
                    continue
                p_text = _clean_whitespace(el.get_text(" ", strip=True))
                if p_text:
                    blocks.append({
                        "section_id": sec_id,
                        "type": "text",
                        "content": p_text,
                    })
                processed_elements.add(el)

        if blocks or heading:
            sections_output.append({
                "section_id": sec_id,
                "heading": heading,
                "blocks": blocks,
            })

    # Calcul du nombre total de mots dans le document extrait
    total_words = 0
    type_counts = {"text": 0, "table": 0, "figure": 0, "equation": 0}

    for section in sections_output:
        for block in section["blocks"]:
            b_type = block["type"]
            type_counts[b_type] = type_counts.get(b_type, 0) + 1
            total_words += len(block["content"].split())

    return {
        "arxiv_id": metadata.get("arxiv_id", ""),
        "clean_id": metadata.get("clean_id", ""),
        "title": metadata.get("title", ""),
        "authors": metadata.get("authors", []),
        "published": metadata.get("published", ""),
        "categories": metadata.get("categories", []),
        "month_stratum": metadata.get("month_stratum", ""),
        "html_url": metadata.get("html_url", ""),
        "total_words": total_words,
        "block_counts": type_counts,
        "sections": sections_output,
    }
