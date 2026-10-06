"""
Module d'orchestration de l'ingestion du corpus arXiv avec échantillonnage
stratifié par mois et tri par pertinence intra-mensuelle.
Gère la redistribution dynamique des quotas en cas de déficit d'un mois,
l'idempotence, la traçabilité des strates et le reporting détaillé.
"""

import json
import os
import statistics
from datetime import datetime, timezone
from typing import Any, Dict, List, Tuple
from tqdm import tqdm

from src.arxiv_client import ArxivClient
from src.parser import parse_paper_html


def _generate_all_months() -> List[Tuple[int, int]]:
    """Génère la liste chronologique des 24 mois pour 2024 et 2025."""
    months = []
    for y in [2024, 2025]:
        for m in range(1, 13):
            months.append((y, m))
    return months


def _compute_initial_quotas(limit: int, months: List[Tuple[int, int]]) -> Dict[str, int]:
    """
    Calcule les quotas initiaux par mois :
    - Si limit <= 24 (mode test), sélectionne `limit` mois répartis régulièrement sur la période.
    - Si limit > 24, répartit équitablement (base + reste).
    """
    num_months = len(months)
    quotas: Dict[str, int] = {}

    if limit <= num_months:
        # Sélection de mois régulièrement espacés sur les deux années
        step = num_months / limit
        selected_indices = set(int(i * step) for i in range(limit))
        for idx, (y, m) in enumerate(months):
            key = f"{y:04d}-{m:02d}"
            quotas[key] = 1 if idx in selected_indices else 0
    else:
        base = limit // num_months
        remainder = limit % num_months
        for idx, (y, m) in enumerate(months):
            key = f"{y:04d}-{m:02d}"
            quotas[key] = base + (1 if idx < remainder else 0)

    return quotas


def run_ingestion(
    limit: int = 300,
    data_dir: str = "data",
    user_agent: str = "rag-portfolio/1.0 (contact: student-portfolio@univ.fr)",
    batch_size: int = 50,
) -> Dict[str, Any]:
    """
    Exécute le pipeline complet avec échantillonnage mensuel stratifié,
    tri par pertinence, et redistribution des quotas non atteints.
    """
    raw_dir = os.path.join(data_dir, "raw")
    processed_dir = os.path.join(data_dir, "processed")
    manifest_path = os.path.join(data_dir, "manifest.json")

    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(processed_dir, exist_ok=True)

    client = ArxivClient(user_agent=user_agent, rate_limit_seconds=3.0)
    all_months = _generate_all_months()
    quotas = _compute_initial_quotas(limit, all_months)

    # 1. Chargement du cache local existant (reprise sur interruption)
    processed_papers: Dict[str, Dict[str, Any]] = {}
    month_counts: Dict[str, int] = {f"{y:04d}-{m:02d}": 0 for y, m in all_months}

    for filename in os.listdir(processed_dir):
        if filename.endswith(".json"):
            filepath = os.path.join(processed_dir, filename)
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    arxiv_id = data.get("arxiv_id")
                    if arxiv_id:
                        stratum = data.get("month_stratum") or data.get("published", "")[:7]
                        processed_papers[arxiv_id] = {
                            "arxiv_id": data["arxiv_id"],
                            "clean_id": data.get("clean_id", ""),
                            "title": data["title"],
                            "authors": data["authors"],
                            "published": data["published"],
                            "categories": data["categories"],
                            "month_stratum": stratum,
                            "html_url": data["html_url"],
                            "total_words": data["total_words"],
                            "block_counts": data.get("block_counts", {}),
                        }
                        if stratum in month_counts:
                            month_counts[stratum] += 1
            except Exception:
                continue

    print(f"[*] Reprise : {len(processed_papers)} article(s) déjà présent(s) dans {processed_dir}")

    shortfalls: List[Dict[str, Any]] = []
    skipped_no_html: List[Dict[str, str]] = []
    skipped_by_month: Dict[str, int] = {f"{y:04d}-{m:02d}": 0 for y, m in all_months}
    accumulated_deficit = 0

    pbar = tqdm(total=limit, initial=min(len(processed_papers), limit), desc="Ingestion stratifiée")

    # 2. Itération sur les 24 mois
    for y, m in all_months:
        if len(processed_papers) >= limit:
            break

        month_key = f"{y:04d}-{m:02d}"
        target_for_month = quotas.get(month_key, 0) + accumulated_deficit
        accumulated_deficit = 0  # Consommé

        if target_for_month <= 0:
            continue

        already_have = month_counts.get(month_key, 0)
        needed = max(0, target_for_month - already_have)

        if needed <= 0:
            continue

        query_str = client.build_monthly_query(y, m)
        start_offset = 0
        month_new_collected = 0

        while month_new_collected < needed:
            batch = client.search_papers(
                query=query_str,
                start=start_offset,
                max_results=batch_size,
                sort_by="relevance",
                sort_order="descending",
                month_stratum=month_key,
            )

            if not batch:
                # Plus d'articles retournés par arXiv pour ce mois
                break

            start_offset += len(batch)

            for paper in batch:
                if month_new_collected >= needed or len(processed_papers) >= limit:
                    break

                arxiv_id = paper["arxiv_id"]
                if arxiv_id in processed_papers:
                    continue

                html_path, _ = client.fetch_html(arxiv_id, raw_dir)

                if not html_path:
                    skipped_by_month[month_key] = skipped_by_month.get(month_key, 0) + 1
                    skipped_no_html.append({
                        "arxiv_id": arxiv_id,
                        "title": paper["title"],
                        "month_stratum": month_key,
                        "reason": "HTML indisponible (code != 200)",
                    })
                    continue

                try:
                    with open(html_path, "r", encoding="utf-8") as f:
                        html_content = f.read()

                    parsed_data = parse_paper_html(html_content, paper)
                    target_json_path = os.path.join(processed_dir, f"{arxiv_id}.json")

                    with open(target_json_path, "w", encoding="utf-8") as f:
                        json.dump(parsed_data, f, ensure_ascii=False, indent=2)

                    paper_entry = {
                        "arxiv_id": parsed_data["arxiv_id"],
                        "clean_id": parsed_data.get("clean_id", ""),
                        "title": parsed_data["title"],
                        "authors": parsed_data["authors"],
                        "published": parsed_data["published"],
                        "categories": parsed_data["categories"],
                        "month_stratum": month_key,
                        "html_url": parsed_data["html_url"],
                        "total_words": parsed_data["total_words"],
                        "block_counts": parsed_data.get("block_counts", {}),
                    }

                    processed_papers[arxiv_id] = paper_entry
                    month_counts[month_key] = month_counts.get(month_key, 0) + 1
                    month_new_collected += 1
                    pbar.update(1)

                except Exception as e:
                    skipped_by_month[month_key] = skipped_by_month.get(month_key, 0) + 1
                    skipped_no_html.append({
                        "arxiv_id": arxiv_id,
                        "title": paper["title"],
                        "month_stratum": month_key,
                        "reason": f"Erreur de parsing : {str(e)}",
                    })

        # Vérification si le quota a été atteint pour ce mois
        total_for_month = month_counts.get(month_key, 0)
        if total_for_month < target_for_month:
            deficit = target_for_month - total_for_month
            accumulated_deficit += deficit
            shortfalls.append({
                "month_stratum": month_key,
                "target": target_for_month,
                "achieved": total_for_month,
                "deficit_redistributed": deficit,
            })

    # 3. Passe de rattrapage éventuelle si déficit résiduel
    if len(processed_papers) < limit and accumulated_deficit > 0:
        print(f"[!] Rattrapage final du déficit ({accumulated_deficit} papier(s)) sur les mois riches...")
        for y, m in reversed(all_months):
            if len(processed_papers) >= limit:
                break
            month_key = f"{y:04d}-{m:02d}"
            query_str = client.build_monthly_query(y, m)
            batch = client.search_papers(
                query=query_str,
                start=50,
                max_results=50,
                sort_by="relevance",
                month_stratum=month_key,
            )
            for paper in batch:
                if len(processed_papers) >= limit:
                    break
                arxiv_id = paper["arxiv_id"]
                if arxiv_id in processed_papers:
                    continue
                html_path, _ = client.fetch_html(arxiv_id, raw_dir)
                if not html_path:
                    skipped_by_month[month_key] = skipped_by_month.get(month_key, 0) + 1
                    continue
                try:
                    with open(html_path, "r", encoding="utf-8") as f:
                        html_content = f.read()
                    parsed_data = parse_paper_html(html_content, paper)
                    target_json_path = os.path.join(processed_dir, f"{arxiv_id}.json")
                    with open(target_json_path, "w", encoding="utf-8") as f:
                        json.dump(parsed_data, f, ensure_ascii=False, indent=2)

                    processed_papers[arxiv_id] = {
                        "arxiv_id": parsed_data["arxiv_id"],
                        "clean_id": parsed_data.get("clean_id", ""),
                        "title": parsed_data["title"],
                        "authors": parsed_data["authors"],
                        "published": parsed_data["published"],
                        "categories": parsed_data["categories"],
                        "month_stratum": month_key,
                        "html_url": parsed_data["html_url"],
                        "total_words": parsed_data["total_words"],
                        "block_counts": parsed_data.get("block_counts", {}),
                    }
                    month_counts[month_key] = month_counts.get(month_key, 0) + 1
                    pbar.update(1)
                except Exception:
                    continue

    pbar.close()

    # 4. Calcul des métriques statistiques (moyenne et médiane)
    total_count = len(processed_papers)
    word_counts = [p["total_words"] for p in processed_papers.values()]
    avg_words = round(statistics.mean(word_counts), 1) if word_counts else 0.0
    median_words = round(statistics.median(word_counts), 1) if word_counts else 0.0

    global_blocks = {"text": 0, "table": 0, "figure": 0, "equation": 0}
    for p in processed_papers.values():
        for b_type, count in p.get("block_counts", {}).items():
            global_blocks[b_type] = global_blocks.get(b_type, 0) + count

    # Tableau récapitulatif mois par mois (24 mois)
    monthly_breakdown = {}
    unmet_months = []
    for y, m in all_months:
        k = f"{y:04d}-{m:02d}"
        achieved = month_counts.get(k, 0)
        target = quotas.get(k, 0)
        ignored = skipped_by_month.get(k, 0)
        status = "Atteint" if achieved >= target else f"Déficit (-{target - achieved})"
        if achieved < target:
            unmet_months.append({"month": k, "target": target, "achieved": achieved})
        monthly_breakdown[k] = {
            "initial_target": target,
            "achieved": achieved,
            "ignored": ignored,
            "status": status,
        }

    # 5. Enregistrement du manifest
    manifest_data = {
        "metadata": {
            "download_timestamp": datetime.now(timezone.utc).isoformat(),
            "target_limit": limit,
            "sampling_strategy": "stratified_monthly_by_relevance",
            "date_range": "2024-01 to 2025-12 (24 months)",
            "query_template": (
                '(cat:cs.CL OR cat:cs.IR) AND '
                '(ti:"retrieval-augmented generation" OR abs:"retrieval-augmented generation") AND '
                'submittedDate:[YYYYMM010000 TO YYYYMMDD2359]'
            ),
            "sort_by": "relevance",
            "sort_order": "descending",
        },
        "stats": {
            "target_limit": limit,
            "total_processed": total_count,
            "total_skipped_no_html": len(skipped_no_html),
            "average_word_count": avg_words,
            "median_word_count": median_words,
            "global_block_distribution": global_blocks,
            "monthly_breakdown_24_months": monthly_breakdown,
            "unmet_quota_months": unmet_months,
        },
        "shortfalls_log": shortfalls,
        "skipped_papers": skipped_no_html,
        "papers": list(processed_papers.values()),
    }

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, ensure_ascii=False, indent=2)

    # 6. Affichage du rapport récapitulatif détaillé
    _print_summary_report(
        limit=limit,
        total_count=total_count,
        skipped_count=len(skipped_no_html),
        avg_words=avg_words,
        median_words=median_words,
        global_blocks=global_blocks,
        all_months=all_months,
        month_counts=month_counts,
        quotas=quotas,
        skipped_by_month=skipped_by_month,
        unmet_months=unmet_months,
    )

    return manifest_data


def _print_summary_report(
    limit: int,
    total_count: int,
    skipped_count: int,
    avg_words: float,
    median_words: float,
    global_blocks: Dict[str, int],
    all_months: List[Tuple[int, int]],
    month_counts: Dict[str, int],
    quotas: Dict[str, int],
    skipped_by_month: Dict[str, int],
    unmet_months: List[Dict[str, Any]],
) -> None:
    """Affiche un tableau complet de 24 lignes avec signalement de tout déficit."""
    print("\n" + "=" * 75)
    print("      RAPPORT D'INGESTION arXiv (ÉCHANTILLONNAGE STRATIFIÉ 24 MOIS)")
    print("=" * 75)
    print(f" Stratégie                    : Mensuelle par pertinence lexicale")
    print(f" Cible demandée               : {limit} papiers avec HTML")
    print(f" Papiers validés et parsés    : {total_count}")
    print(f" Papiers ignorés (sans HTML)  : {skipped_count}")
    print(f" Nombre moyen de mots         : {avg_words:,.1f} mots")
    print(f" Nombre médian de mots        : {median_words:,.1f} mots")
    print("-" * 75)
    print(" Répartition temporelle détaillée (24 mois) :")
    print(f"  {'Mois':<9} | {'Obtenus / Quota':<17} | {'Ignorés':<9} | {'Statut'}")
    print("  " + "-" * 70)
    for y, m in all_months:
        k = f"{y:04d}-{m:02d}"
        achieved = month_counts.get(k, 0)
        target = quotas.get(k, 0)
        ignored = skipped_by_month.get(k, 0)
        status = "OK" if achieved >= target else f"Déficit (-{target - achieved})"
        print(f"  {k:<9} | {achieved:>3} / {target:>2}            | {ignored:>3}       | {status}")
    print("-" * 75)
    if unmet_months:
        print(" [!] MOIS N'AYANT PAS ATTEINT LEUR QUOTA INITIAL :")
        for u in unmet_months:
            print(f"  - {u['month']} : {u['achieved']}/{u['target']} (déficit redistribué)")
    else:
        print(" [OK] Tous les 24 mois ont atteint ou dépassé leur quota initial.")
    print("-" * 75)
    print(" Répartition des blocs extraits :")
    print(f"  - Paragraphes de texte     : {global_blocks.get('text', 0)}")
    print(f"  - Tableaux (Markdown)      : {global_blocks.get('table', 0)}")
    print(f"  - Légendes de figures      : {global_blocks.get('figure', 0)}")
    print(f"  - Équations hors-bloc      : {global_blocks.get('equation', 0)}")
    print("=" * 75 + "\n")
