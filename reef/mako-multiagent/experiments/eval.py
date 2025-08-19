# experiments/eval.py (pure‑python Rouge + helpers)
import re
from typing import Tuple

def normalize_text(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[^a-z0-9\s]", "", s)
    return s


def string_accuracy(pred: str, gold: str) -> int:
    return int(normalize_text(pred) == normalize_text(gold))

# Minimal Rouge‑N and Rouge‑L (F1) implementations

def _ngrams(tokens, n):
    return [tuple(tokens[i:i+n]) for i in range(0, max(0, len(tokens)-n+1))]


def rouge_n(pred: str, gold: str, n: int = 1) -> float:
    p = normalize_text(pred).split()
    g = normalize_text(gold).split()
    if not p or not g:
        return 0.0
    pn = _ngrams(p, n)
    gn = _ngrams(g, n)
    overlap = sum(1 for t in pn if t in gn)
    if overlap == 0:
        return 0.0
    prec = overlap / max(1, len(pn))
    rec = overlap / max(1, len(gn))
    if prec + rec == 0:
        return 0.0
    return 2 * prec * rec / (prec + rec)


def rouge_l(pred: str, gold: str) -> float:
    p = normalize_text(pred).split()
    g = normalize_text(gold).split()
    # LCS length
    dp = [[0]*(len(g)+1) for _ in range(len(p)+1)]
    for i in range(1, len(p)+1):
        for j in range(1, len(g)+1):
            dp[i][j] = dp[i-1][j-1]+1 if p[i-1]==g[j-1] else max(dp[i-1][j], dp[i][j-1])
    lcs = dp[-1][-1]
    if lcs == 0:
        return 0.0
    prec = lcs / max(1, len(p))
    rec = lcs / max(1, len(g))
    return 2 * prec * rec / (prec + rec)


def rouge_scores(pred: str, gold: str) -> Tuple[float, float, float]:
    return rouge_n(pred, gold, 1), rouge_n(pred, gold, 2), rouge_l(pred, gold)