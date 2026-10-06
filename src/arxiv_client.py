"""
Module de communication avec l'API arXiv et de téléchargement des versions HTML.
Respecte scrupuleusement le rate limiting imposé par arXiv (>= 3 secondes entre requêtes)
et gère les requêtes mensuelles stratifiées triées par pertinence.
"""

import calendar
import os
import time
import urllib.parse
import xml.etree.ElementTree as ET
from typing import Dict, List, Optional, Tuple
import requests


class ArxivClient:
    """Client HTTP pour l'API Atom d'arXiv et la récupération de HTML expérimental."""

    BASE_API_URL = "http://export.arxiv.org/api/query"
    ATOM_NS = {"atom": "http://www.w3.org/2005/Atom"}

    def __init__(
        self,
        user_agent: str = "rag-portfolio/1.0 (contact: student-portfolio@univ.fr)",
        rate_limit_seconds: float = 3.0,
    ):
        self.user_agent = user_agent
        self.rate_limit_seconds = rate_limit_seconds
        self.last_request_time = 0.0
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": self.user_agent})

    def _throttle(self) -> None:
        """
        Garantit qu'au moins `rate_limit_seconds` se sont écoulées
        depuis la dernière requête HTTP adressée aux serveurs d'arXiv.
        """
        elapsed = time.time() - self.last_request_time
        if elapsed < self.rate_limit_seconds:
            time.sleep(self.rate_limit_seconds - elapsed)
        self.last_request_time = time.time()

    def build_monthly_query(self, year: int, month: int) -> str:
        """
        Construit l'expression de recherche exacte pour un mois donné :
        - Expression exacte "retrieval-augmented generation" dans titre OU résumé
        - Catégories cs.CL ou cs.IR
        - Plage temporelle stricte pour le mois donné (YYYYMM010000 à YYYYMMDD2359)
        """
        last_day = calendar.monthrange(year, month)[1]
        start_date = f"{year}{month:02d}010000"
        end_date = f"{year}{month:02d}{last_day:02d}2359"

        return (
            '(cat:cs.CL OR cat:cs.IR) AND '
            '(ti:"retrieval-augmented generation" OR abs:"retrieval-augmented generation") AND '
            f'submittedDate:[{start_date} TO {end_date}]'
        )

    def search_papers(
        self,
        query: str,
        start: int = 0,
        max_results: int = 50,
        sort_by: str = "relevance",
        sort_order: str = "descending",
        month_stratum: Optional[str] = None,
    ) -> List[Dict]:
        """
        Interroge l'API Atom d'arXiv et renvoie une liste de métadonnées.
        Par défaut, tri par pertinence pour éviter le biais temporel intra-mensuel.
        """
        self._throttle()

        params = {
            "search_query": query,
            "start": start,
            "max_results": max_results,
            "sortBy": sort_by,
            "sortOrder": sort_order,
        }

        encoded_query = urllib.parse.urlencode(params)
        url = f"{self.BASE_API_URL}?{encoded_query}"

        # Gestion des micro-coupures et ReadTimeout fréquents sur export.arxiv.org
        max_retries = 3
        response = None
        for attempt in range(max_retries):
            try:
                self._throttle()
                response = self.session.get(url, timeout=35)
                response.raise_for_status()
                break
            except requests.RequestException as e:
                if attempt < max_retries - 1:
                    wait_time = 5.0 * (attempt + 1)
                    print(f"\n[!] Latence réseau arXiv ({e}). Nouvel essai dans {wait_time}s ({attempt + 1}/{max_retries})...")
                    time.sleep(wait_time)
                else:
                    print(f"\n[!] Échec de requête API arXiv après {max_retries} essais : {e}")
                    return []

        if response is None:
            return []

        try:
            root = ET.fromstring(response.text)
        except ET.ParseError:
            return []

        entries = root.findall("atom:entry", self.ATOM_NS)
        papers = []

        for entry in entries:
            id_url = entry.find("atom:id", self.ATOM_NS)
            if id_url is None or not id_url.text:
                continue

            raw_id = id_url.text.strip().split("/abs/")[-1]
            clean_id = raw_id.split("v")[0] if "v" in raw_id else raw_id

            title_el = entry.find("atom:title", self.ATOM_NS)
            title = (
                " ".join(title_el.text.split())
                if title_el is not None and title_el.text
                else "Untitled"
            )

            published_el = entry.find("atom:published", self.ATOM_NS)
            published = published_el.text.strip() if published_el is not None else ""

            summary_el = entry.find("atom:summary", self.ATOM_NS)
            summary = (
                " ".join(summary_el.text.split())
                if summary_el is not None and summary_el.text
                else ""
            )

            authors = []
            for author_el in entry.findall("atom:author", self.ATOM_NS):
                name_el = author_el.find("atom:name", self.ATOM_NS)
                if name_el is not None and name_el.text:
                    authors.append(name_el.text.strip())

            categories = []
            for cat_el in entry.findall("atom:category", self.ATOM_NS):
                term = cat_el.get("term")
                if term:
                    categories.append(term)

            papers.append({
                "arxiv_id": raw_id,
                "clean_id": clean_id,
                "title": title,
                "published": published,
                "authors": authors,
                "categories": categories,
                "summary": summary,
                "html_url": f"https://arxiv.org/html/{raw_id}",
                "month_stratum": month_stratum or (published[:7] if len(published) >= 7 else "unknown"),
            })

        return papers

    def fetch_html(self, arxiv_id: str, raw_dir: str) -> Tuple[Optional[str], bool]:
        """
        Télécharge la version HTML d'un article et la stocke dans raw_dir.
        Reprise sur interruption : réutilise le fichier si déjà présent.
        """
        os.makedirs(raw_dir, exist_ok=True)
        target_path = os.path.join(raw_dir, f"{arxiv_id}.html")

        if os.path.exists(target_path) and os.path.getsize(target_path) > 0:
            return target_path, False

        html_url = f"https://arxiv.org/html/{arxiv_id}"
        max_retries = 2

        for attempt in range(max_retries):
            self._throttle()
            try:
                resp = self.session.get(html_url, timeout=30)
                if resp.status_code == 200 and "<html" in resp.text.lower():
                    with open(target_path, "w", encoding="utf-8") as f:
                        f.write(resp.text)
                    return target_path, True
                else:
                    return None, False
            except requests.RequestException:
                if attempt < max_retries - 1:
                    time.sleep(3.0)
                else:
                    return None, False

        return None, False
