"""Analyse quantitative de l'impact des commandes LaTeX sur les passages textuels.
Mesures requises pour l'Étape 4 :
1. 400 caractères au milieu des 3 premiers passages textuels les plus longs en tokens.
2. Statistiques des commandes LaTeX (\\[a-zA-Z]+) sur les 11 430 passages textuels :
   - Médiane, P90, P99
   - Passages avec >= 20 et >= 50 commandes
   - Part des 328 passages > 512 tokens qui en contiennent >= 20
   - Concentration par papier (Top 10 papiers)
   - Ratio médian tokens/mots avec et sans >= 20 commandes
3. Longueur moyenne en tokens BM25 des passages >= 20 commandes vs autres.
4. Évaluation d'une règle de nettoyage conservatrice pour indexing_text().
"""

import json
import re
from collections import Counter
import sys
from pathlib import Path

# Ajout de la racine du projet au PYTHONPATH
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from transformers import AutoTokenizer
from src.tokenizer import ScientificTokenizer

PASSAGES_PATH = Path("data/passages.jsonl")
MODEL_NAME = "BAAI/bge-small-en-v1.5"

def clean_latex_conservative(text: str) -> str:
    """Règle de nettoyage conservatrice pour indexing_text :
    - Supprime la mise en forme sans contenu sémantique :
      \\color[...]{...}, \\color{...}, \\left, \\right, espacements (\\,, \\;, \\!, \\quad, \\qquad, \\enspace)
    - Déballe les commandes de fonte/style en conservant leur contenu textuel/mathématique :
      \\mathbf{X} -> X, \\bm{X} -> X, \\mathrm{X} -> X, \\text{X} -> X, \\textbf{X} -> X, \\mathit{X} -> X, \\mathcal{X} -> X
    """
    # 1. Suppression des balises de couleur \color[rgb]{...} ou \color{...}
    t = re.sub(r'\\color(?:\[[^\]]*\])?\{[^\}]*\}', '', text)
    # 2. Suppression des délimiteurs d'échelle \left et \right
    t = re.sub(r'\\(?:left|right)\b', '', t)
    # 3. Suppression des commandes d'espacement LaTeX
    t = re.sub(r'\\(?:,|;|!|quad|qquad|enspace)\b', ' ', t)
    # 4. Déballage récursif simple des commandes de style \cmd{contenu}
    # (ex: \bm{q} -> q, \mathbf{W} -> W, \mathcal{D} -> D)
    style_cmds = r'(?:mathbf|bm|mathrm|text|textbf|mathit|mathcal|mathsf|boldsymbol)'
    # On applique 3 passes pour gérer l'imbrication éventuelle (ex: \bm{\mathcal{D}})
    for _ in range(3):
        t = re.sub(rf'\\{style_cmds}\{{([^{{}}]*)\}}', r'\1', t)
    # 5. Normalisation des espaces multiples créés par les suppressions
    t = re.sub(r'[ \t]+', ' ', t)
    return t

def main():
    print("Chargement des passages et tokenizers...")
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    bm25_tok = ScientificTokenizer()

    with open(PASSAGES_PATH, "r", encoding="utf-8") as f:
        passages = [json.loads(line) for line in f]

    text_passages = [p for p in passages if p["type"] == "text"]
    id2p = {p["passage_id"]: p for p in text_passages}
    print(f"Total passages de type 'text' : {len(text_passages)}")

    # 1. 400 caractères bruts pris au MILIEU pour les 3 premiers
    top3_ids = ["2411.07773_p0007", "2411.07773_p0011", "2411.07773_p0010"]
    print("\n" + "=" * 60)
    print("1. EXTRAITS BRUTS DU MILIEU (400 CARACTÈRES)")
    print("=" * 60)
    for pid in top3_ids:
        p = id2p[pid]
        txt = p["content"]
        mid = len(txt) // 2
        start = max(0, mid - 200)
        end = min(len(txt), mid + 200)
        snippet = txt[start:end]
        print(f"\n--- {pid} (Doc: {p.get('doc_id')}, Section: {p.get('section_path')}, Mots: {len(txt.split())}) ---")
        print(snippet)

    # 2. Analyse des commandes LaTeX
    cmd_pattern = re.compile(r'\\[a-zA-Z]+')

    cmd_counts = []
    tokens_bge_before = []
    tokens_bge_after = []
    words_counts = []
    bm25_tokens_before = []
    paper_counter_ge20 = Counter()

    for p in text_passages:
        txt = p["content"]
        cmds = cmd_pattern.findall(txt)
        n_cmds = len(cmds)
        cmd_counts.append(n_cmds)
        w = len(txt.split())
        words_counts.append(w)

        # BGE tokens brut
        tb_before = len(tok.encode(txt, add_special_tokens=True))
        tokens_bge_before.append(tb_before)

        # BGE tokens après nettoyage conservateur
        txt_clean = clean_latex_conservative(txt)
        tb_after = len(tok.encode(txt_clean, add_special_tokens=True))
        tokens_bge_after.append(tb_after)

        # BM25 tokens brut
        t_bm25 = len(bm25_tok.tokenize(txt))
        bm25_tokens_before.append(t_bm25)

        doc_id = p.get("doc_id") or p["passage_id"].split("_")[0]
        if n_cmds >= 20:
            paper_counter_ge20[doc_id] += 1

    cmd_counts = np.array(cmd_counts)
    tokens_bge_before = np.array(tokens_bge_before)
    tokens_bge_after = np.array(tokens_bge_after)
    words_counts = np.array(words_counts)
    bm25_tokens_before = np.array(bm25_tokens_before)

    print("\n" + "=" * 60)
    print("2. STATISTIQUES DES COMMANDES LATEX SUR 11 430 PASSAGES TEXT")
    print("=" * 60)
    print(f"Médiane des commandes LaTeX / passage : {np.median(cmd_counts):.1f}")
    print(f"P90 des commandes LaTeX / passage     : {np.percentile(cmd_counts, 90):.1f}")
    print(f"P99 des commandes LaTeX / passage     : {np.percentile(cmd_counts, 99):.1f}")
    
    n_ge20 = np.sum(cmd_counts >= 20)
    n_ge50 = np.sum(cmd_counts >= 50)
    print(f"Passages avec >= 20 commandes LaTeX   : {n_ge20} ({n_ge20 / len(text_passages) * 100:.2f}%)")
    print(f"Passages avec >= 50 commandes LaTeX   : {n_ge50} ({n_ge50 / len(text_passages) * 100:.2f}%)")

    over_512_before = np.sum(tokens_bge_before > 512)
    over_512_and_ge20 = np.sum((tokens_bge_before > 512) & (cmd_counts >= 20))
    print(f"\nTotal passages text > 512 tokens BGE  : {over_512_before}")
    print(f"Dont contenant >= 20 commandes LaTeX  : {over_512_and_ge20} ({over_512_and_ge20 / over_512_before * 100:.2f}%)")

    print(f"\nConcentration par papier (Top 10 papiers avec >= 20 commandes, total {n_ge20} passages) :")
    for rank, (doc_id, cnt) in enumerate(paper_counter_ge20.most_common(10), 1):
        print(f"  {rank:2d}. {doc_id} : {cnt} passages ({cnt / n_ge20 * 100:.2f}% du total >= 20)")

    ratios = tokens_bge_before / np.maximum(words_counts, 1)
    print(f"\nRatio tokens/mots médian global                : {np.median(ratios):.2f}")
    print(f"Ratio tokens/mots médian (>= 20 commandes LaTeX) : {np.median(ratios[cmd_counts >= 20]):.2f}")
    print(f"Ratio tokens/mots médian (< 20 commandes LaTeX)  : {np.median(ratios[cmd_counts < 20]):.2f}")

    print("\n" + "=" * 60)
    print("3. LONGUEUR MOYENNE EN TOKENS BM25")
    print("=" * 60)
    mean_bm25_ge20 = np.mean(bm25_tokens_before[cmd_counts >= 20])
    median_bm25_ge20 = np.median(bm25_tokens_before[cmd_counts >= 20])
    mean_bm25_lt20 = np.mean(bm25_tokens_before[cmd_counts < 20])
    median_bm25_lt20 = np.median(bm25_tokens_before[cmd_counts < 20])
    print(f"Passages >= 20 commandes : Moyenne = {mean_bm25_ge20:.2f} tokens | Médiane = {median_bm25_ge20:.1f} tokens")
    print(f"Passages <  20 commandes : Moyenne = {mean_bm25_lt20:.2f} tokens | Médiane = {median_bm25_lt20:.1f} tokens")

    print("\n" + "=" * 60)
    print("4. SIMULATION DE GAIN AVEC indexing_text() CONSERVATEUR")
    print("=" * 60)
    over_512_after = np.sum(tokens_bge_after > 512)
    print(f"Passages text > 512 tokens AVANT nettoyage : {over_512_before} (2.87%)")
    print(f"Passages text > 512 tokens APRÈS nettoyage : {over_512_after} ({over_512_after / len(text_passages) * 100:.2f}%)")
    print(f"Gain net : {over_512_before - over_512_after} passages sauvés de la troncature (-{(over_512_before - over_512_after) / over_512_before * 100:.1f}%)")
    print(f"Longueur maximale BGE tokens : {np.max(tokens_bge_before)} -> {np.max(tokens_bge_after)}")

if __name__ == "__main__":
    main()
