"""
Point d'entrée CLI pour le découpage en passages (chunking) du corpus arXiv.

Usage standard :
    python run_chunking.py

Options configurables :
    python run_chunking.py --overlap 0
    python run_chunking.py --target-words 250 --max-words 350 --min-words 60
"""

import argparse
import sys
from src.chunker import chunk_corpus


def main():
    parser = argparse.ArgumentParser(
        description="Découpage en passages sémantiques (chunking) du corpus arXiv traité."
    )
    parser.add_argument(
        "--processed-dir",
        type=str,
        default="data/processed",
        help="Dossier contenant les fichiers JSON traités (défaut: data/processed)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/passages.jsonl",
        help="Fichier de sortie au format JSONL (défaut: data/passages.jsonl)",
    )
    parser.add_argument(
        "--target-words",
        type=int,
        default=250,
        help="Nombre de mots cible par passage textuel (défaut: 250)",
    )
    parser.add_argument(
        "--max-words",
        type=int,
        default=350,
        help="Plafond strict de mots par passage textuel (défaut: 350)",
    )
    parser.add_argument(
        "--min-words",
        type=int,
        default=60,
        help="Seuil minimum de mots pour la fusion intra-section (défaut: 60)",
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=40,
        help="Chevauchement en mots entre passages textuels consécutifs (défaut: 40, option: 0)",
    )

    args = parser.parse_args()

    try:
        report = chunk_corpus(
            processed_dir=args.processed_dir,
            output_file=args.output,
            target_words=args.target_words,
            max_words=args.max_words,
            min_words=args.min_words,
            overlap_words=args.overlap,
        )
        if report.get("cross_section_errors", 0) > 0:
            print("[ERREUR] Des erreurs d'isolation inter-sections ont été détectées !")
            sys.exit(1)
        sys.exit(0)
    except Exception as e:
        print(f"[ERREUR] Échec lors du chunking : {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
