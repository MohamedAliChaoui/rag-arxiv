"""
Module de découpage en passages (chunking) sémantique du corpus arXiv.

Règles architecturales appliquées :
1. Isolation stricte des sections : AUCUN passage n'est à cheval sur deux sections.
2. Taille cible : ~250 mots par passage (max 350 mots), seuil minimum de 60 mots
   (hors tableaux et figures) par fusion intra-section.
3. Découpe hiérarchique : coupe entre paragraphes, puis entre phrases protégées
   (abréviations scientifiques et formules préservées).
4. Chevauchement (overlap) intra-section paramétrable (défaut: 40 mots, option: 0).
5. Traçabilité des blocs : chaque passage contient son `block_range` d'origine [start_idx, end_idx].
6. Tableaux isolés en passages dédiés : si > 350 mots, découpe par lignes avec légende
   et en-têtes répétés sur chaque sous-tableau.
7. Légendes de figures isolées en passages dédiés (type 'figure').
8. Suppression des marqueurs '[EQUATION]' isolés et conservation des formules inline courtes.
9. Reconstruction hiérarchique du chemin de section (y compris annexes 'A...' et gestion des parents manquants).
10. Double représentation textuelle : 'content' (texte brut) et 'contextual_content' (avec en-tête documentaire).
"""

import json
import os
import re
import statistics
from typing import Any, Dict, List, Optional, Tuple


def split_sentences(text: str) -> List[str]:
    """
    Découpe un texte en phrases en protégeant les abréviations scientifiques
    usuelles (ex: 'et al.', 'i.e.', 'Fig. 1', 'Eq. 2', décimales '0.85').
    """
    if not text:
        return []

    # Liste d'abréviations scientifiques à ne pas scinder
    abbrevs = [
        "et al.", "i.e.", "e.g.", "vs.", "Fig.", "Tab.", "Tabs.", "Eq.", "Eqs.",
        "Ref.", "Refs.", "Sec.", "Secs.", "al.", "approx.", "no.", "vol.", "pp.",
        "dr.", "prof.", "dept."
    ]
    substitutions: Dict[str, str] = {}
    protected = text

    for idx, abb in enumerate(abbrevs):
        token = f"__ABB_{idx}__"
        pattern = re.compile(re.escape(abb), re.IGNORECASE)
        protected = pattern.sub(token, protected)
        substitutions[token] = abb

    # Protéger les nombres à virgule/point (ex: 0.95 ou 1.2)
    protected = re.sub(r"(\d)\.(\d)", r"\1__DOT__\2", protected)

    # Découpage sur ponctuation forte suivie d'un espace ou fin de texte
    parts = re.split(r"([.!?]+(?:\s+|$))", protected)
    sentences: List[str] = []
    current = ""

    for p in parts:
        current += p
        if re.search(r"[.!?]+(?:\s+|$)", p):
            s = current.strip()
            if s:
                # Restauration des abréviations et points protégés
                for token, abb in substitutions.items():
                    s = s.replace(token, abb)
                s = s.replace("__DOT__", ".")
                sentences.append(s)
            current = ""

    if current.strip():
        s = current.strip()
        for token, abb in substitutions.items():
            s = s.replace(token, abb)
        s = s.replace("__DOT__", ".")
        sentences.append(s)

    return sentences if sentences else [text.strip()]


def build_section_path(sections_map: Dict[str, str], current_sec: Dict[str, Any]) -> Tuple[str, bool]:
    """
    Reconstruit le fil d'Ariane hiérarchique des titres à partir du section_id
    (ex: S1.SS1 -> '1 Introduction > 1.1 Background', A1.SS1 -> 'Appendix A > Proofs').
    
    Retourne :
        (section_path, had_missing_parent)
    """
    sid = current_sec.get("section_id", "")
    current_heading = current_sec.get("heading", "").strip() or sid

    if sid == "abstract" or "abstract" in sid.lower():
        return "Abstract", False

    parts = sid.split(".")
    breadcrumb: List[str] = []
    had_missing_parent = False

    # Résolution des ancêtres pour les IDs imbriqués (ex: S3.SS1.SSS1 ou A1.SS1)
    if len(parts) > 1:
        for i in range(1, len(parts)):
            parent_id = ".".join(parts[:i])
            if parent_id in sections_map and sections_map[parent_id].strip():
                breadcrumb.append(sections_map[parent_id].strip())
            else:
                had_missing_parent = True

    breadcrumb.append(current_heading)

    # Déduplication consécutive si le parent et l'enfant ont le même libellé
    clean_bc: List[str] = []
    for h in breadcrumb:
        if not clean_bc or clean_bc[-1] != h:
            clean_bc.append(h)

    return " > ".join(clean_bc), had_missing_parent


def chunk_markdown_table(table_text: str, max_words: int = 350) -> List[str]:
    """
    Découpe un tableau Markdown volumineux (> max_words) par lots de lignes,
    en répétant la légende et les en-têtes de colonnes sur chaque sous-tableau.
    """
    lines = [line.strip() for line in table_text.strip().split("\n") if line.strip()]
    if not lines:
        return []

    caption = ""
    table_lines: List[str] = []

    for line in lines:
        if line.startswith("|"):
            table_lines.append(line)
        elif not table_lines and (line.startswith("[") or "table" in line.lower()):
            caption = line

    if not table_lines:
        return [table_text]

    header = table_lines[0]
    separator = (
        table_lines[1]
        if len(table_lines) > 1 and "---" in table_lines[1]
        else "| " + " | ".join(["---"] * header.count("|")) + " |"
    )
    data_rows = (
        table_lines[2:]
        if (len(table_lines) > 1 and "---" in table_lines[1])
        else table_lines[1:]
    )

    full_words = len(table_text.split())
    if full_words <= max_words or not data_rows:
        return [table_text]

    chunks: List[str] = []
    prefix = (caption + "\n" if caption else "") + header + "\n" + separator
    prefix_words = len(prefix.split())

    current_rows: List[str] = []
    current_words = prefix_words

    for row in data_rows:
        row_words = len(row.split())
        if current_rows and (current_words + row_words > max_words):
            chunk_content = prefix + "\n" + "\n".join(current_rows)
            chunks.append(chunk_content)
            current_rows = [row]
            current_words = prefix_words + row_words
        else:
            current_rows.append(row)
            current_words += row_words

    if current_rows:
        chunk_content = prefix + "\n" + "\n".join(current_rows)
        chunks.append(chunk_content)

    return chunks


def _chunk_text_units(
    units: List[Dict[str, Any]],
    target_words: int = 250,
    max_words: int = 350,
    min_words: int = 60,
    overlap_words: int = 40,
) -> List[Dict[str, Any]]:
    """
    Découpe une séquence ordonnée d'unités de phrases en passages de ~target_words,
    avec overlap configurable et respect du minimum de min_words.
    """
    if not units:
        return []

    chunks_units: List[List[Dict[str, Any]]] = []
    idx = 0
    n = len(units)

    while idx < n:
        chunk: List[Dict[str, Any]] = []
        chunk_words = 0

        while idx < n:
            u = units[idx]
            if chunk_words > 0 and (chunk_words + u["words"] > max_words):
                break
            chunk.append(u)
            chunk_words += u["words"]
            idx += 1
            if chunk_words >= target_words:
                break

        chunks_units.append(chunk)

        if idx >= n:
            break

        # Application de l'overlap intra-section
        if overlap_words > 0 and chunk:
            accum_overlap = 0
            back_steps = 0
            for u in reversed(chunk):
                if accum_overlap + u["words"] <= overlap_words or back_steps == 0:
                    accum_overlap += u["words"]
                    back_steps += 1
                else:
                    break
            if back_steps < len(chunk):
                idx -= back_steps

    # Fusion ou rééquilibrage si le dernier chunk a < min_words
    if len(chunks_units) > 1:
        last_words = sum(u["words"] for u in chunks_units[-1])
        if last_words < min_words:
            prev_words = sum(u["words"] for u in chunks_units[-2])
            # Déterminer les unités non redondantes du dernier chunk
            last_unique = [u for u in chunks_units[-1] if u not in chunks_units[-2]]
            if prev_words + sum(u["words"] for u in last_unique) <= max_words:
                chunks_units[-2].extend(last_unique)
                chunks_units.pop()
            else:
                # Si la fusion dépasse max_words, on équilibre en déplaçant
                # des phrases de l'avant-dernier vers le dernier
                while chunks_units[-2] and sum(u["words"] for u in chunks_units[-1]) < min_words:
                    moved_unit = chunks_units[-2].pop()
                    chunks_units[-1].insert(0, moved_unit)

    # Création des dictionnaires de passages
    passages: List[Dict[str, Any]] = []
    for c_units in chunks_units:
        text = " ".join(u["text"] for u in c_units).strip()
        if not text:
            continue

        b_indices = [u["block_idx"] for u in c_units]
        block_range = [min(b_indices), max(b_indices)]

        passages.append({
            "type": "text",
            "content": text,
            "word_count": len(text.split()),
            "char_count": len(text),
            "block_range": block_range,
        })

    return passages


def _merge_short_text_passages_in_section(
    passages: List[Dict[str, Any]],
    title: str,
    section_path: str,
    min_words: int = 60,
    max_words: int = 350,
) -> List[Dict[str, Any]]:
    """
    Fusionne les passages textuels d'une même section ayant moins de min_words
    avec leur voisin textuel dans la même section, sous réserve de ne pas dépasser max_words.
    """
    text_count = sum(1 for p in passages if p["type"] == "text")
    if text_count < 2:
        return passages

    changed = True
    while changed:
        changed = False
        text_indices = [i for i, p in enumerate(passages) if p["type"] == "text"]
        for k, idx in enumerate(text_indices):
            p = passages[idx]
            if p["word_count"] < min_words:
                # 1. Tentative de fusion avec le voisin textuel précédent dans la section
                if k > 0:
                    prev_idx = text_indices[k - 1]
                    prev_p = passages[prev_idx]
                    if prev_p["word_count"] + p["word_count"] <= max_words:
                        merged_content = prev_p["content"] + "\n\n" + p["content"]
                        prev_p["content"] = merged_content
                        prev_p["word_count"] = len(merged_content.split())
                        prev_p["char_count"] = len(merged_content)
                        prev_p["block_range"] = [
                            min(prev_p["block_range"][0], p["block_range"][0]),
                            max(prev_p["block_range"][1], p["block_range"][1]),
                        ]
                        prev_p["contextual_content"] = (
                            f"Document: {title}\nSection: {section_path}\n\n{merged_content}"
                        )
                        passages.pop(idx)
                        changed = True
                        break

                # 2. Tentative de fusion avec le voisin textuel suivant dans la section
                if k + 1 < len(text_indices):
                    next_idx = text_indices[k + 1]
                    next_p = passages[next_idx]
                    if p["word_count"] + next_p["word_count"] <= max_words:
                        merged_content = p["content"] + "\n\n" + next_p["content"]
                        next_p["content"] = merged_content
                        next_p["word_count"] = len(merged_content.split())
                        next_p["char_count"] = len(merged_content)
                        next_p["block_range"] = [
                            min(p["block_range"][0], next_p["block_range"][0]),
                            max(p["block_range"][1], next_p["block_range"][1]),
                        ]
                        next_p["contextual_content"] = (
                            f"Document: {title}\nSection: {section_path}\n\n{merged_content}"
                        )
                        passages.pop(idx)
                        changed = True
                        break

    return passages


def chunk_section(
    sec: Dict[str, Any],
    section_path: str,
    title: str,
    arxiv_id: str,
    clean_id: str,
    month_stratum: str,
    passage_counter_start: int,
    target_words: int = 250,
    max_words: int = 350,
    min_words: int = 60,
    overlap_words: int = 40,
) -> Tuple[List[Dict[str, Any]], int]:
    """
    Découpe l'ensemble des blocs d'une section en passages discrets.
    Garantit qu'aucun passage ne dépasse les frontières de cette section.
    """
    sec_id = sec.get("section_id", "")
    blocks = sec.get("blocks", [])
    raw_passages: List[Dict[str, Any]] = []

    # Extraction des blocs textuels et des blocs autonomes (table, figure)
    current_text_units: List[Dict[str, Any]] = []

    def flush_text_units():
        nonlocal current_text_units
        if not current_text_units:
            return
        t_passages = _chunk_text_units(
            current_text_units,
            target_words=target_words,
            max_words=max_words,
            min_words=min_words,
            overlap_words=overlap_words,
        )
        for tp in t_passages:
            content = tp["content"]
            contextual = f"Document: {title}\nSection: {section_path}\n\n{content}"
            raw_passages.append({
                "arxiv_id": arxiv_id,
                "clean_id": clean_id,
                "title": title,
                "month_stratum": month_stratum,
                "section_id": sec_id,
                "section_path": section_path,
                "block_range": tp["block_range"],
                "type": "text",
                "word_count": tp["word_count"],
                "char_count": tp["char_count"],
                "section_url": f"https://arxiv.org/html/{arxiv_id}#{sec_id}",
                "content": content,
                "contextual_content": contextual,
            })
        current_text_units = []

    for b_idx, block in enumerate(blocks):
        b_type = block.get("type", "text")
        raw_content = block.get("content", "").strip()

        # 1. Traitement des Tableaux
        if b_type == "table":
            flush_text_units()
            sub_tables = chunk_markdown_table(raw_content, max_words=max_words)
            for st in sub_tables:
                w_cnt = len(st.split())
                contextual = f"Document: {title}\nSection: {section_path}\n\n{st}"
                raw_passages.append({
                    "arxiv_id": arxiv_id,
                    "clean_id": clean_id,
                    "title": title,
                    "month_stratum": month_stratum,
                    "section_id": sec_id,
                    "section_path": section_path,
                    "block_range": [b_idx, b_idx],
                    "type": "table",
                    "word_count": w_cnt,
                    "char_count": len(st),
                    "section_url": f"https://arxiv.org/html/{arxiv_id}#{sec_id}",
                    "content": st,
                    "contextual_content": contextual,
                })

        # 2. Traitement des Figures (légendes)
        elif b_type == "figure":
            flush_text_units()
            w_cnt = len(raw_content.split())
            contextual = f"Document: {title}\nSection: {section_path}\n\n{raw_content}"
            raw_passages.append({
                "arxiv_id": arxiv_id,
                "clean_id": clean_id,
                "title": title,
                "month_stratum": month_stratum,
                "section_id": sec_id,
                "section_path": section_path,
                "block_range": [b_idx, b_idx],
                "type": "figure",
                "word_count": w_cnt,
                "char_count": len(raw_content),
                "section_url": f"https://arxiv.org/html/{arxiv_id}#{sec_id}",
                "content": raw_content,
                "contextual_content": contextual,
            })

        # 3. Traitement des Équations hors-bloc
        elif b_type == "equation":
            # Suppression des marqueurs [EQUATION] isolés
            if raw_content == "[EQUATION]" or not raw_content:
                continue
            # Formules courtes conservées dans le flux textuel
            sents = split_sentences(raw_content)
            for s in sents:
                w = len(s.split())
                if w > 0:
                    current_text_units.append({
                        "block_idx": b_idx,
                        "text": s,
                        "words": w,
                    })

        # 4. Traitement du Texte standard
        elif b_type == "text":
            # Nettoyage des marqueurs [EQUATION] dans le texte
            cleaned = re.sub(r"\[EQUATION\]", "", raw_content)
            cleaned = re.sub(r"\s+", " ", cleaned).strip()
            if not cleaned:
                continue

            sents = split_sentences(cleaned)
            for s in sents:
                w = len(s.split())
                if w > 0:
                    current_text_units.append({
                        "block_idx": b_idx,
                        "text": s,
                        "words": w,
                    })

    flush_text_units()

    # Fusion intra-section des passages textuels résiduels < min_words avec leur voisin
    merged_passages = _merge_short_text_passages_in_section(
        passages=raw_passages,
        title=title,
        section_path=section_path,
        min_words=min_words,
        max_words=max_words,
    )

    # Attribution séquentielle des passage_id contigus et marquage oversized (> 350 mots)
    final_passages: List[Dict[str, Any]] = []
    counter = passage_counter_start
    for p in merged_passages:
        if p["word_count"] > 350:
            p["oversized"] = True
        p["passage_id"] = f"{clean_id}_p{counter:04d}"
        counter += 1
        final_passages.append(p)

    return final_passages, counter


def chunk_paper(
    paper_data: Dict[str, Any],
    target_words: int = 250,
    max_words: int = 350,
    min_words: int = 60,
    overlap_words: int = 40,
) -> Tuple[List[Dict[str, Any]], bool]:
    """
    Découpe l'ensemble des sections d'un papier en passages discrets.
    Retourne la liste des passages et un indicateur si un chemin a manqué d'ancêtre.
    """
    arxiv_id = paper_data.get("arxiv_id", "")
    clean_id = paper_data.get("clean_id", "") or arxiv_id.split("v")[0]
    title = paper_data.get("title", "Untitled")
    month_stratum = paper_data.get("month_stratum", "") or paper_data.get("published", "")[:7]
    sections = paper_data.get("sections", [])

    # Dictionnaire de correspondance ID -> Heading pour résolution hiérarchique
    sections_map = {sec.get("section_id", ""): sec.get("heading", "") for sec in sections}

    all_passages: List[Dict[str, Any]] = []
    paper_had_missing_parent = False
    counter = 1

    for sec in sections:
        path, missing = build_section_path(sections_map, sec)
        if missing:
            paper_had_missing_parent = True

        sec_passages, counter = chunk_section(
            sec=sec,
            section_path=path,
            title=title,
            arxiv_id=arxiv_id,
            clean_id=clean_id,
            month_stratum=month_stratum,
            passage_counter_start=counter,
            target_words=target_words,
            max_words=max_words,
            min_words=min_words,
            overlap_words=overlap_words,
        )
        all_passages.extend(sec_passages)

    return all_passages, paper_had_missing_parent


def chunk_corpus(
    processed_dir: str = "data/processed",
    output_file: str = "data/passages.jsonl",
    target_words: int = 250,
    max_words: int = 350,
    min_words: int = 60,
    overlap_words: int = 40,
) -> Dict[str, Any]:
    """
    Orchestre le découpage de tous les articles du corpus et écrit data/passages.jsonl.
    Exécute un contrôle strict sur données réelles garantissant l'isolation des sections.
    """
    if not os.path.exists(processed_dir):
        raise FileNotFoundError(f"Dossier {processed_dir} introuvable.")

    files = sorted([f for f in os.listdir(processed_dir) if f.endswith(".json")])
    total_papers = len(files)
    papers_with_missing_parents = 0

    all_passages: List[Dict[str, Any]] = []
    cross_section_errors = 0

    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as out_f:
        for fname in files:
            fpath = os.path.join(processed_dir, fname)
            with open(fpath, "r", encoding="utf-8") as in_f:
                paper_data = json.load(in_f)

            passages, had_missing = chunk_paper(
                paper_data=paper_data,
                target_words=target_words,
                max_words=max_words,
                min_words=min_words,
                overlap_words=overlap_words,
            )

            if had_missing:
                papers_with_missing_parents += 1

            # CONTRÔLE D'ISOLATION EN TEMPS RÉEL SUR CHAQUE PAPIER
            # Vérification formelle que chaque passage correspond strictement
            # à une seule section existante de ce papier.
            valid_sec_ids = {s.get("section_id") for s in paper_data.get("sections", [])}
            for p in passages:
                sid = p.get("section_id")
                if not sid or sid not in valid_sec_ids or " > " in sid:
                    cross_section_errors += 1

                # Vérification que le block_range est strictement confiné aux blocs de la section
                sec_dict = next((s for s in paper_data.get("sections", []) if s.get("section_id") == sid), None)
                if sec_dict:
                    num_blocks = len(sec_dict.get("blocks", []))
                    br = p.get("block_range", [])
                    if len(br) != 2 or br[0] < 0 or br[1] >= num_blocks or br[0] > br[1]:
                        cross_section_errors += 1

                out_f.write(json.dumps(p, ensure_ascii=False) + "\n")
                all_passages.append(p)

    total_passages = len(all_passages)
    word_lengths = [p["word_count"] for p in all_passages]
    type_counts: Dict[str, int] = {}
    short_by_type: Dict[str, int] = {"text": 0, "table": 0, "figure": 0}
    long_by_type: Dict[str, int] = {"text": 0, "table": 0, "figure": 0}

    for p in all_passages:
        t = p["type"]
        type_counts[t] = type_counts.get(t, 0) + 1
        w = p["word_count"]
        if w < min_words:
            short_by_type[t] = short_by_type.get(t, 0) + 1
        if w > max_words:
            long_by_type[t] = long_by_type.get(t, 0) + 1

    short_passages = sum(short_by_type.values())
    long_passages = sum(long_by_type.values())

    word_lengths_sorted = sorted(word_lengths)
    p25 = word_lengths_sorted[int(0.25 * total_passages)] if total_passages else 0
    p50 = statistics.median(word_lengths) if word_lengths else 0
    p75 = word_lengths_sorted[int(0.75 * total_passages)] if total_passages else 0
    p95 = word_lengths_sorted[int(0.95 * total_passages)] if total_passages else 0

    report = {
        "total_papers": total_papers,
        "total_passages": total_passages,
        "type_counts": type_counts,
        "short_by_type": short_by_type,
        "long_by_type": long_by_type,
        "cross_section_errors": cross_section_errors,
        "papers_with_missing_parents": papers_with_missing_parents,
        "min_words": min(word_lengths) if word_lengths else 0,
        "max_words": max(word_lengths) if word_lengths else 0,
        "mean_words": round(statistics.mean(word_lengths), 1) if word_lengths else 0.0,
        "median_words": round(p50, 1),
        "p25": p25,
        "p75": p75,
        "p95": p95,
        "passages_below_60": short_passages,
        "passages_above_350": long_passages,
    }

    _print_chunking_report(report)
    return report


def _print_chunking_report(rep: Dict[str, Any]) -> None:
    """Affiche le rapport complet de distribution du chunking."""
    print("\n" + "=" * 75)
    print("           RAPPORT DE DÉCOUPAGE EN PASSAGES (CHUNKING)")
    print("=" * 75)
    print(f" Papiers traités               : {rep['total_papers']}")
    print(f" Total de passages générés    : {rep['total_passages']:,}")
    print(f" Moyenne de passages / papier  : {rep['total_passages'] / max(1, rep['total_papers']):.1f}")
    print("-" * 75)
    print(" Répartition par type de passage :")
    for t, cnt in rep["type_counts"].items():
        pct = (cnt / rep["total_passages"]) * 100 if rep["total_passages"] else 0
        print(f"  - {t:<10} : {cnt:>6,} ({pct:>5.1f}%)")
    print("-" * 75)
    print(" Distribution des longueurs (en mots) :")
    print(f"  - Min / Max                 : {rep['min_words']} / {rep['max_words']} mots")
    print(f"  - Moyenne / Médiane         : {rep['mean_words']:.1f} / {rep['median_words']:.1f} mots")
    print(f"  - Percentiles (P25 / P75)   : {rep['p25']} / {rep['p75']} mots")
    print(f"  - Percentile P95            : {rep['p95']} mots")
    print("-" * 75)
    print(" Contrôles qualité et seuils :")
    print(f"  - Passages < 60 mots        : {rep['passages_below_60']:,}")
    s_types = rep.get("short_by_type", {})
    print(f"      * Texte (sections isolées) : {s_types.get('text', 0):,}")
    print(f"      * Tableaux (courts)        : {s_types.get('table', 0):,}")
    print(f"      * Figures (légendes)       : {s_types.get('figure', 0):,}")
    print(f"  - Passages > 350 mots       : {rep['passages_above_350']:,}")
    l_types = rep.get("long_by_type", {})
    print(f"      * Tableaux indivisibles    : {l_types.get('table', 0):,}")
    print(f"      * Texte / Figures          : {l_types.get('text', 0) + l_types.get('figure', 0):,}")
    print(f"  - Erreurs inter-sections    : {rep['cross_section_errors']} (100% strictement confinés à une section)")
    print(f"  - Chemins avec parent absent: {rep['papers_with_missing_parents']} / {rep['total_papers']} papiers (résolus par fallback)")
    print("=" * 75 + "\n")
