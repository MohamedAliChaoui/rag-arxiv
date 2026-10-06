"""
Point d'entrée CLI pour lancer l'ingestion du corpus arXiv.

Usage :
    python run_ingestion.py --limit 5
    python run_ingestion.py --limit 300
"""

import argparse
import sys
from src.ingestion import run_ingestion


def main():
    parser = argparse.ArgumentParser(
        description="Ingestion et structuration des articles arXiv pour RAG."
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Nombre cible d'articles valides avec HTML à télécharger et parser (défaut: 5)",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="Répertoire racine pour le stockage des données (défaut: data)",
    )

    args = parser.parse_args()

    try:
        run_ingestion(limit=args.limit, data_dir=args.data_dir)
    except KeyboardInterrupt:
        print("\n[!] Ingestion interrompue par l'utilisateur. L'état actuel a été sauvegardé.")
        sys.exit(0)


if __name__ == "__main__":
    main()
