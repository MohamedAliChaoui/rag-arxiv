"""Script de mesure réelle du débit d'encodage dense sur 200 passages réels.

Modèle : BAAI/bge-small-en-v1.5
Matériel : CPU
Compare le débit réel sur les 200 passages réels avec 4 threads puis 8 threads.
"""

import json
import os
import random
import time
from pathlib import Path

PASSAGES_PATH = Path("data/passages.jsonl")
MODEL_NAME = "BAAI/bge-small-en-v1.5"
N_SAMPLE = 200
BATCH_SIZE = 32
SEED = 42

def get_process_memory_mb() -> float:
    try:
        import psutil
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / (1024 * 1024)
    except ImportError:
        return 0.0

def main():
    print(f"Chargement de {N_SAMPLE} passages depuis {PASSAGES_PATH} (graine {SEED})...")
    with open(PASSAGES_PATH, "r", encoding="utf-8") as f:
        all_passages = [json.loads(line) for line in f]

    random.seed(SEED)
    sample_passages = random.sample(all_passages, N_SAMPLE)
    sample_texts = [p["content"] for p in sample_passages]

    total_corpus_count = len(all_passages)
    print(f"Total passages dans le corpus : {total_corpus_count}")
    print(f"Taille de l'échantillon : {len(sample_texts)}")

    # Tri par longueur pour optimiser le padding dans les batchs
    indexed_texts = list(enumerate(sample_texts))
    indexed_texts.sort(key=lambda x: len(x[1]))
    sorted_texts = [t for _, t in indexed_texts]

    print(f"\nChargement du modèle SentenceTransformer: {MODEL_NAME}...")
    t0_load = time.perf_counter()
    from sentence_transformers import SentenceTransformer
    import torch
    
    device = "cpu"
    model = SentenceTransformer(MODEL_NAME, device=device)
    load_time = time.perf_counter() - t0_load
    print(f"Modèle chargé en {load_time:.2f} s.")

    # Warmup
    _ = model.encode(["Warmup text to initialize weights and cache."], device=device, normalize_embeddings=True)

    results = {}
    for n_threads in [4, 8]:
        torch.set_num_threads(n_threads)
        print(f"\n--- Mesure avec {n_threads} threads PyTorch sur les {N_SAMPLE} passages réels ---")
        mem_before = get_process_memory_mb()
        t0 = time.perf_counter()
        _ = model.encode(
            sorted_texts,
            batch_size=BATCH_SIZE,
            show_progress_bar=True,
            normalize_embeddings=True,
            device=device,
        )
        elapsed = time.perf_counter() - t0
        mem_after = get_process_memory_mb()
        throughput = N_SAMPLE / elapsed
        est_total_min = (total_corpus_count / throughput) / 60.0

        results[n_threads] = {
            "elapsed": elapsed,
            "throughput": throughput,
            "est_min": est_total_min,
            "ram_mb": mem_after,
        }
        print(f"-> {n_threads} threads : {elapsed:.2f} s ({throughput:.2f} passages/s) | Estimation totale : {est_total_min:.1f} min")

    print("\n" + "=" * 60)
    print("COMPARAISON DU DÉBIT D'ENCODAGE SUR 200 PASSAGES RÉELS")
    print("=" * 60)
    for th, r in results.items():
        print(f"Threads: {th:2d} | Temps: {r['elapsed']:5.2f} s | Débit: {r['throughput']:4.2f} pass/s | Est. totale (16 310 pass): {r['est_min']:4.1f} min")
    diff_pct = (results[4]["elapsed"] - results[8]["elapsed"]) / results[4]["elapsed"] * 100
    print(f"\nÉcart mesuré (8 vs 4 threads) : {diff_pct:+.1f}% de temps d'encodage.")
    print("=" * 60)

if __name__ == "__main__":
    main()
