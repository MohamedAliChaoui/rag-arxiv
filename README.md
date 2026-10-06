# RAG Hybride sur Articles arXiv

Projet de portfolio (Master 2 Informatique - Spécialisation IA).

Ce projet implémente un système de **Retrieval-Augmented Generation (RAG)** complet et évalué rigoureusement sur un corpus de 300 à 500 articles scientifiques arXiv spécialisés en NLP et Retrieval.

---

## Architecture visée

1. **Ingestion & Découpage** : Collecte via l'API arXiv, nettoyage du texte brut, découpage en passages sémantiques avec métadonnées (titre, auteurs, section, identifiant de passage).
2. **Retrieval Hybride** :
   - Retrieval lexical : BM25.
   - Retrieval dense : Embeddings & recherche vectorielle (FAISS).
   - Fusion hybride : Reciprocal Rank Fusion (RRF) ou combinaison pondérée normalisée.
3. **Reranking** : Reranker cross-encoder pour réordonner les meilleurs candidats.
4. **Génération avec Citations & Refus** :
   - Citations explicites rattachées aux identifiants de passage.
   - Mécanisme de refus de répondre si les passages récupérés ne contiennent pas l'information (anti-hallucination).
5. **Évaluation Rigoureuse** :
   - Évaluation du Retrieval : Recall@k, MRR, nDCG.
   - Évaluation de la Génération : Fidélité contextuelle (groundedness), exactitude des citations.

---

## Installation & Démarrage

### 1. Prérequis
- Python 3.10+ (développé avec Python 3.13)
- Git

### 2. Environnement virtuel

```bash
# Création de l'environnement virtuel
python -m venv .venv

# Activation sous Windows (PowerShell)
.venv\Scripts\Activate.ps1

# Activation sous Linux / macOS
source .venv/bin/activate
```

### 3. Installation des dépendances

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## Structure du projet

```text
├── src/          # Modules Python du pipeline (ingestion, retrieval, generation)
├── data/         # Données brutes et traitées (ignoré par Git)
├── eval/         # Scripts de benchmarking et métriques d'évaluation
├── notebooks/    # Analyses exploratoires et prototypes
├── requirements.txt
└── README.md
```

---

## Statut du projet
- [x] **Étape 0** : Structure du projet, environnement et `.gitignore`.
- [ ] **Étape 1** : Téléchargement et nettoyage des papiers arXiv.
