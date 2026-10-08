"""
Module de tokenisation scientifique et technique pour le retrieval BM25.

Spécifications appliquées :
1. Prise en charge des accents et caractères Unicode :
   Utilisation de la classe de caractères Unicode `[^\\W_]` (lettres et chiffres de tout alphabet)
   au lieu de `[A-Za-z0-9]`, garantissant la préservation intégrale des accents français/européens.
2. Mots composés scientifiques (ex: 'Self-RAG', 'retrieval-augmented', 'BERT-base') :
   Émission du mot composé entier ET de chacune de ses sous-parties élémentaires
   (ex: 'self-rag' -> ['self-rag', 'self', 'rag']).
3. Termes techniques et métriques :
   Préservation des suffixes de métriques et paramètres (ex: 'nDCG@10', 'MRR@5').
4. Limite documentée sur 'GPT-3.5-Turbo' :
   La coexistence d'un point de version ('3.5') et de tirets ('-') est traitée en considérant
   le point comme délimiteur de ponctuation hors nombres isolés. 'GPT-3.5-Turbo' est ainsi
   segmenté en deux sous-composés ['gpt-3', 'gpt', '3', '5-turbo', '5', 'turbo'].
   L'inclusion aveugle du point dans le motif de mot composé risquerait en effet d'agréger
   des fins de phrases sans espace ('fin.Début').
5. Options configurables :
   - Minuscules (lowercase : bool, défaut True).
   - Stopwords anglais (remove_stopwords : bool, défaut False).
   - Racinisation (stemming : bool, défaut False, implémentation Porter Stemmer autonome).
"""

from dataclasses import dataclass, asdict
import re
from typing import Any, Dict, List, Optional, Set


# Stopwords anglais usuels pour l'information retrieval (33 mots standards BM25)
DEFAULT_ENGLISH_STOPWORDS: Set[str] = {
    "a", "about", "an", "and", "are", "as", "at", "be", "by", "for",
    "from", "how", "in", "is", "it", "of", "on", "or", "that", "the",
    "this", "to", "was", "what", "when", "where", "which", "who", "will",
    "with", "the", "were", "their"
}


class PorterStemmer:
    """
    Implémentation légère et autonome de l'algorithme de racinisation de Porter (1980)
    pour la langue anglaise, ne nécessitant aucune dépendance C binaire.
    """
    def __init__(self):
        self.b = ""
        self.k = 0
        self.k0 = 0
        self.j = 0

    def _cons(self, i: int) -> bool:
        if self.b[i] in "aeiou":
            return False
        if self.b[i] == "y":
            if i == self.k0:
                return True
            else:
                return not self._cons(i - 1)
        return True

    def _m(self) -> int:
        n = 0
        i = self.k0
        while True:
            if i > self.j:
                return n
            if not self._cons(i):
                break
            i += 1
        i += 1
        while True:
            while True:
                if i > self.j:
                    return n
                if self._cons(i):
                    break
                i += 1
            i += 1
            n += 1
            while True:
                if i > self.j:
                    return n
                if not self._cons(i):
                    break
                i += 1
            i += 1

    def _vowelinstem(self) -> bool:
        for i in range(self.k0, self.j + 1):
            if not self._cons(i):
                return True
        return False

    def _doublec(self, i: int) -> bool:
        if i < self.k0 + 1:
            return False
        if self.b[i] != self.b[i - 1]:
            return False
        return self._cons(i)

    def _cvc(self, i: int) -> bool:
        if i < self.k0 + 2 or not self._cons(i) or self._cons(i - 1) or not self._cons(i - 2):
            return False
        ch = self.b[i]
        if ch in "wxy":
            return False
        return True

    def _ends(self, s: str) -> bool:
        length = len(s)
        o = self.k - length + 1
        if o < self.k0:
            return False
        if self.b[o : self.k + 1] != s:
            return False
        self.j = self.k - length
        return True

    def _setto(self, s: str):
        length = len(s)
        o = self.j + 1
        self.b = self.b[:o] + s + self.b[o + length :]
        self.k = self.j + length

    def _r(self, s: str):
        if self._m() > 0:
            self._setto(s)

    def _step1ab(self):
        if self.b[self.k] == "s":
            if self._ends("sses"):
                self.k -= 2
            elif self._ends("ies"):
                self._setto("i")
            elif self.b[self.k - 1] != "s":
                self.k -= 1
        if self._ends("eed"):
            if self._m() > 0:
                self.k -= 1
        elif (self._ends("ed") or self._ends("ing")) and self._vowelinstem():
            self.k = self.j
            if self._ends("at"):
                self._setto("ate")
            elif self._ends("bl"):
                self._setto("ble")
            elif self._ends("iz"):
                self._setto("ize")
            elif self._doublec(self.k):
                self.k -= 1
                ch = self.b[self.k]
                if ch in "lsz":
                    self.k += 1
            elif self._m() == 1 and self._cvc(self.k):
                self._setto("e")

    def _step1c(self):
        if self._ends("y") and self._vowelinstem():
            self.b = self.b[: self.k] + "i" + self.b[self.k + 1 :]

    def _step2(self):
        # Simplification standard de l'étape 2
        suffixes = {
            "ational": "ate", "tional": "tion", "enci": "ence", "anci": "ance",
            "izer": "ize", "bli": "ble", "alli": "al", "entli": "ent", "eli": "e",
            "ousli": "ous", "ization": "ize", "ation": "ate", "ator": "ate",
            "alism": "al", "iveness": "ive", "fulness": "ful", "ousness": "ous",
            "aliti": "al", "iviti": "ive", "biliti": "ble"
        }
        for suff, rep in suffixes.items():
            if self._ends(suff):
                if self._m() > 0:
                    self._setto(rep)
                break

    def stem(self, word: str) -> str:
        if len(word) <= 2:
            return word
        self.b = word
        self.k = len(word) - 1
        self.k0 = 0
        self._step1ab()
        self._step1c()
        self._step2()
        return self.b[: self.k + 1]


@dataclass
class TokenizerConfig:
    """
    Configuration du pipeline de tokenisation BM25.
    """
    lowercase: bool = True
    preserve_compound: bool = True
    remove_stopwords: bool = False
    stem: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TokenizerConfig":
        return cls(
            lowercase=d.get("lowercase", True),
            preserve_compound=d.get("preserve_compound", True),
            remove_stopwords=d.get("remove_stopwords", False),
            stem=d.get("stem", False),
        )


class ScientificTokenizer:
    """
    Tokenizer adapté aux documents scientifiques en NLP et Retrieval (corpus arXiv).
    """

    def __init__(self, config: Optional[TokenizerConfig] = None):
        self.config = config or TokenizerConfig()
        self._stemmer = PorterStemmer() if self.config.stem else None
        self._stopwords = DEFAULT_ENGLISH_STOPWORDS

        # Motif Regex Unicode :
        # 1. Mots composés avec tirets internes (ex: Self-RAG, retrieval-augmented)
        # 2. Métriques avec suffixe numérique @ (ex: nDCG@10, MRR@5)
        # 3. Mots simples ou nombres isolés avec support des accents [^\W_]+
        self._pattern = re.compile(
            r"[^\W_]+(?:-[^\W_]+)+|[^\W_]+@\d+|[^\W_]+"
        )

    def tokenize(self, text: str) -> List[str]:
        """
        Découpe un texte en liste de jetons selon la configuration active.
        Pour un mot composé, émet le composé ET chacune de ses sous-parties.
        """
        if not text:
            return []

        processed_text = text.lower() if self.config.lowercase else text
        tokens: List[str] = []

        for match in self._pattern.finditer(processed_text):
            raw_token = match.group(0)

            if "-" in raw_token and self.config.preserve_compound:
                # Émission du mot composé complet
                tokens.append(raw_token)
                # Émission des sous-parties élémentaires
                subparts = [p for p in raw_token.split("-") if p]
                tokens.extend(subparts)
            else:
                tokens.append(raw_token)

        # Filtrage optionnel des stopwords
        if self.config.remove_stopwords:
            tokens = [t for t in tokens if t not in self._stopwords]

        # Racinisation optionnelle (stemming)
        if self.config.stem and self._stemmer:
            tokens = [self._stemmer.stem(t) for t in tokens]

        return tokens


def tokenize(text: str, config: Optional[TokenizerConfig] = None) -> List[str]:
    """Fonction utilitaire directe pour tokeniser un texte."""
    tokenizer = ScientificTokenizer(config)
    return tokenizer.tokenize(text)
