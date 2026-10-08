"""Script de mesure réelle du débit d'encodage dense sur 200 passages.

Modèle : BAAI/bge-small-en-v1.5
Matériel : CPU
Mesure :
  1. Chargement du modèle et tokenizer
  2. Échantillon de 200 passages (graine 42)
  3. Encodage avec batchs triés par longueur (batch_size=32)
  4. Débit réel (passages/seconde) et extrapolation sur 16 310 passages
  5. Pic mémoire RAM du processus
"""

import json
import os
import random
import time
import tracemalloc
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
        # Fallback via tracemalloc si psutil non disponible
        return 0.0

def main():
    print(f"Chargement de {N_SAMPLE} passages depuis {PASSAGES_PATH} (graine {SEED})...")
    with open(PASSAGES_PATH, "r", encoding="utf-8") as f:
        all_passages = [json.loads(line) for line in f]

    random.seed(SEED)
    sample_passages = random.sample(all_passages, N_SAMPLE)
    sample_texts = [p["content"] for p in sample_passages]

    print(f"Total passages dans le corpus : {len(all_passages)}")
    print(f"Taille de l'échantillon : {len(sample_texts)}")

    # Mesure mémoire avant import/chargement
    mem_before = get_process_memory_mb()

    print(f"\nChargement du modèle SentenceTransformer: {MODEL_NAME}...")
    t0_load = time.perf_counter()
    from sentence_transformers import SentenceTransformer
    import torch
    
    # Configuration CPU explicite
    device = "cpu"
    model = SentenceTransformer(MODEL_NAME, device=device)
    load_time = time.perf_counter() - t0_load
    print(f"Modèle chargé en {load_time:.2f} s sur {device}.")

    # Warmup sur 1 phrase
    _ = model.encode(["Warmup text to initialize weights and cache."], device=device, normalize_embeddings=True)

    # Tri par longueur pour optimiser le padding dans les batchs
    # On garde les indices pour simuler le comportement réel
    indexed_texts = list(enumerate(sample_texts))
    indexed_texts.sort(key=lambda x: len(x[1]))
    sorted_texts = [t for _, t in indexed_texts]

    mem_before_encode = get_process_memory_mb()

    print(f"\nEncodage de {N_SAMPLE} passages (batch_size={BATCH_SIZE}, tri par longueur)...")
    t0_encode = time.perf_counter()
    embeddings = model.encode(
        sorted_texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        normalize_embeddings=True,
        device=device
    )
    encode_time = time.perf_counter() - t0_encode

    mem_after_encode = get_process_memory_mb()

    throughput = N_SAMPLE / encode_time
    total_corpus_count = len(all_passages)
    estimated_total_sec = total_corpus_count / throughput
    estimated_total_min = estimated_total_sec / 60.0

    print("\n" + "=" * 50)
    print("RÉSULTATS DE LA MESURE DU DÉBIT")
    print("=" * 50)
    print(f"Modèle                 : {MODEL_NAME}")
    print(f"Périphérique           : CPU ({torch.get_num_threads()} threads)")
    print(f"Taille échantillon     : {N_SAMPLE} passages")
    print(f"Temps d'encodage mesuré: {encode_time:.2f} s")
    print(f"Débit réel             : {throughput:.2f} passages/seconde")
    print(f"Estimation corpus ({total_corpus_count} passages) : {estimated_total_sec:.1f} s (~{estimated_total_min:.1f} minutes)")
    if mem_after_encode > 0:
        print(f"Mémoire RAM du process : {mem_after_encode:.1f} MB (delta: +{mem_after_encode - mem_before:.1f} MB)")
    print(f"Forme des embeddings   : {embeddings.shape} (dtype: {embeddings.dtype})")
    print("=" * 50)

if __name__ == "__main__":
    main()
