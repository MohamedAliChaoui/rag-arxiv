"""
Module définissant l'interface commune des retrievers pour le projet RAG.
Cette interface abstraite est respectée par :
- BM25Retriever (Étape 3 : lexical)
- DenseRetriever (Étape 4 : dense embeddings + FAISS)
- HybridRetriever (Étape 5 : fusion hybride RRF)
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class SearchResult:
    """
    Résultat d'une recherche unitaire retourné par un retriever.
    
    Attributs :
        passage_id : Identifiant unique du passage (ex: '2401.01511_p0001').
        score : Score de pertinence (score BM25, similarité cosinus ou score fusionné).
        rank : Rang du résultat (1-indexé : 1, 2, ..., k).
        metadata : Métadonnées optionnelles (titre, section_path, extrait, etc.).
    """
    passage_id: str
    score: float
    rank: int
    metadata: Optional[Dict[str, Any]] = None


class BaseRetriever(ABC):
    """
    Contrat abstrait commun pour tous les moteurs de recherche du pipeline.
    """

    @abstractmethod
    def search(self, query: str, k: int = 10) -> List[SearchResult]:
        """
        Recherche les k passages les plus pertinents pour une requête donnée.

        Args:
            query : Texte de la requête utilisateur.
            k : Nombre maximal de résultats souhaités (défaut : 10).

        Returns:
            Liste de SearchResult ordonnée par score décroissant (rang 1 à k).
        """
        pass
