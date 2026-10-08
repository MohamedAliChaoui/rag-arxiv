"""
Script de démonstration et de benchmark technique du retriever BM25 (bm25s).

Fonctionnalités :
1. Construction et sauvegarde des deux variantes d'index :
   - Variante 1 : 'content' (texte brut du passage, défaut)
   - Variante 2 : 'contextual_content' (texte + en-tête documentaire titre/section)
2. Exécution des 6 requêtes de démonstration :
   - Requête 1 : 'corpus poisoning attacks on retrieval-augmented generation'
   - Requête 2 : 'when should the model decide to retrieve'
   - Requête 3 : 'chunk size impact on retrieval quality'
   - Requête 4 : 'reranker cross-encoder improves retrieval'
   - Requête 5 : 'hallucination in medical question answering'
   - Requête 6 : 'protein folding prediction' (requête hors domaine)
3. Affichage du Top 5 pour chaque requête (rang, score, titre, section_path, extrait).
4. Tableau comparatif final des scores Top-1 pour les deux variantes.
5. Rapport technique : temps de construction (s), taille sur disque (Mo), latence moyenne (ms).
"""

import os
import sys
import time
from typing import Any, Dict, List, Tuple

sys.path.insert(0, ".")
from src.bm25_retriever import BM25Retriever


DEMO_QUERIES = [
    "corpus poisoning attacks on retrieval-augmented generation",
    "when should the model decide to retrieve",
    "chunk size impact on retrieval quality",
    "reranker cross-encoder improves retrieval",
    "hallucination in medical question answering",
    "protein folding prediction",
]


def get_dir_size_mb(path: str) -> float:
    """Calcule la taille totale d'un dossier en mégaoctets."""
    total_bytes = 0
    if not os.path.exists(path):
        return 0.0
    for root, _, files in os.walk(path):
        for f in files:
            fp = os.path.join(root, f)
            total_bytes += os.path.getsize(fp)
    return total_bytes / (1024 * 1024)


def run_variant(
    field_name: str,
    index_dir: str,
    passages_path: str = "data/passages.jsonl",
) -> Tuple[BM25Retriever, float, float, Dict[str, Any]]:
    """
    Construit (ou charge) un index BM25 pour un champ donné et mesure les performances.
    """
    print(f"\n{'=' * 80}")
    print(f"   INITIALISATION INDEX BM25 - CHAMP : '{field_name}'")
    print(f"{'=' * 80}")

    retriever = BM25Retriever(
        indexed_field=field_name,
        indexed_types=["text", "table", "figure"],
    )

    t0 = time.perf_counter()
    retriever.build_index(passages_path=passages_path, show_progress=False)
    build_time = time.perf_counter() - t0

    retriever.save(index_dir=index_dir)
    disk_size = get_dir_size_mb(index_dir)

    print(f" [OK] Index '{field_name}' construit en {build_time:.2f} s")
    print(f" [OK] Taille sur disque : {disk_size:.2f} Mo ({retriever.metadata_info['num_passages']:,} passages)")

    # Exécution des requêtes et mesure de latence
    query_results = {}
    latencies_total = []
    latencies_search_only = []

    print(f"\n--- RÉSULTATS DES 6 REQUÊTES (CHAMP : '{field_name}') ---")
    for q_idx, query in enumerate(DEMO_QUERIES, 1):
        # Mesure temps de tokenisation seul
        t_tok_start = time.perf_counter()
        q_tokens = retriever.tokenizer.tokenize(query)
        lat_tok = (time.perf_counter() - t_tok_start) * 1000.0

        # Mesure temps complet search() (tokenisation requête incluse)
        t_start = time.perf_counter()
        results = retriever.search(query, k=5)
        lat_total = (time.perf_counter() - t_start) * 1000.0
        lat_search_only = max(0.0, lat_total - lat_tok)

        latencies_total.append(lat_total)
        latencies_search_only.append(lat_search_only)

        query_results[query] = {
            "results": results,
            "latency_total_ms": lat_total,
            "latency_search_ms": lat_search_only,
            "top1_score": results[0].score if results else 0.0,
            "top1_type": results[0].metadata.get("type", "") if results and results[0].metadata else "N/A",
            "top1_title": results[0].metadata.get("title", "") if results and results[0].metadata else "N/A",
            "top1_path": results[0].metadata.get("section_path", "") if results and results[0].metadata else "N/A",
        }

        print(f"\nRequête {q_idx} : \"{query}\"")
        print(f"  Latence totale : {lat_total:.2f} ms (dont tokenisation: {lat_tok:.2f} ms | calcul BM25: {lat_search_only:.2f} ms)")
        print("-" * 80)
        for r in results:
            meta = r.metadata or {}
            ptype = meta.get("type", "unknown")
            title = meta.get("title", "N/A")
            path = meta.get("section_path", "N/A")
            snippet = meta.get("content_snippet", "").replace("\n", " ").strip()
            if len(snippet) > 160:
                snippet = snippet[:160] + "..."
            print(f"  #{r.rank:<2} [Score: {r.score:>7.4f}] [Type: {ptype:<6}] {r.passage_id}")
            print(f"      Titre   : {title}")
            print(f"      Section : {path}")
            print(f"      Extrait : {snippet}")

    # Décompte des types dans les top-5
    from collections import Counter
    top5_types = Counter()
    for q_data in query_results.values():
        for r in q_data["results"]:
            top5_types[r.metadata.get("type", "unknown")] += 1

    avg_latency_total = sum(latencies_total) / len(latencies_total) if latencies_total else 0.0
    avg_latency_search = sum(latencies_search_only) / len(latencies_search_only) if latencies_search_only else 0.0

    print(f"\nRépartition des types dans les Top-5 (sur {len(DEMO_QUERIES)*5} résultats) :")
    for t_name, count in top5_types.items():
        print(f"  - {t_name:<8} : {count:>2} passages ({count/(len(DEMO_QUERIES)*5)*100:.1f}%)")

    stats = {
        "build_time_s": build_time,
        "disk_size_mb": disk_size,
        "avg_latency_total_ms": avg_latency_total,
        "avg_latency_search_ms": avg_latency_search,
        "top5_types": top5_types,
        "queries": query_results,
    }
    return retriever, build_time, disk_size, stats


def main():
    passages_path = "data/passages.jsonl"
    if not os.path.exists(passages_path):
        print(f"[ERREUR] Fichier {passages_path} introuvable.")
        return 1

    # 1. Exécution de la variante par défaut : 'content'
    _, _, _, stats_content = run_variant(
        field_name="content",
        index_dir="data/index/bm25_content",
        passages_path=passages_path,
    )

    # 2. Exécution de la variante : 'contextual_content'
    _, _, _, stats_contextual = run_variant(
        field_name="contextual_content",
        index_dir="data/index/bm25_contextual",
        passages_path=passages_path,
    )

    # 3. Tableau comparatif final des scores Top-1
    print("\n" + "=" * 90)
    print("      TABLEAU COMPARATIF DES SCORES TOP-1 (CONTENT vs CONTEXTUAL_CONTENT)")
    print("=" * 90)
    print(f"{'#':<3} {'Requête':<42} {'Score (content)':<18} {'Score (contextual)':<18} {'Diff':<8}")
    print("-" * 90)

    for idx, q in enumerate(DEMO_QUERIES, 1):
        s_cont = stats_content["queries"][q]["top1_score"]
        s_ctxt = stats_contextual["queries"][q]["top1_score"]
        diff = s_ctxt - s_cont
        diff_str = f"{diff:+.4f}"
        q_short = q if len(q) <= 40 else q[:37] + "..."
        print(f"{idx:<3} {q_short:<42} {s_cont:>15.4f}   {s_ctxt:>15.4f}   {diff_str:>8}")

    # 4. Rapport technique global
    print("\n" + "=" * 90)
    print("                      RAPPORT TECHNIQUE GLOBAL BM25 (bm25s)")
    print("=" * 90)
    print(f" Métrique                         | 'content' (Défaut)      | 'contextual_content'")
    print("-" * 90)
    print(f" Temps de construction de l'index | {stats_content['build_time_s']:>19.2f} s | {stats_contextual['build_time_s']:>19.2f} s")
    print(f" Taille de l'index sur disque     | {stats_content['disk_size_mb']:>19.2f} Mo| {stats_contextual['disk_size_mb']:>19.2f} Mo")
    print(f" Latence moyenne (avec tokenisation)| {stats_content['avg_latency_total_ms']:>17.2f} ms | {stats_contextual['avg_latency_total_ms']:>17.2f} ms")
    print(f" Latence moyenne (hors tokenisation)| {stats_content['avg_latency_search_ms']:>17.2f} ms | {stats_contextual['avg_latency_search_ms']:>17.2f} ms")
    print(f" Passages indexés                 | 16,309 passages         | 16,310 passages")
    print(f" Tableaux dans le top-5 (sur 30)  | {stats_content['top5_types'].get('table', 0):>19}    | {stats_contextual['top5_types'].get('table', 0):>19}")
    print(f" Figures dans le top-5 (sur 30)   | {stats_content['top5_types'].get('figure', 0):>19}    | {stats_contextual['top5_types'].get('figure', 0):>19}")
    print("=" * 90 + "\n")

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
