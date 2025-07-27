#!/usr/bin/env python3
"""
Compute 
1.  Level-text accuracy  
2.  Score accuracy  
3.  RMSE  
4.  Pearson correlation  
5.  Spearman correlation   

for a JSON-lines file whose objects contain
    "response": "8. …",
    "labels":   "7. …"
"""

import argparse
import json
import math
import numpy as np
import re
from pathlib import Path
from scipy.stats import pearsonr, spearmanr


def extract_score_level(text: str):
    """
    Split a string of the form '8. some description'
    into (score:int, level:str).  Returns (None, None)
    if the pattern is missing.
    """
    m = re.match(r"\s*(\d+)\s*\.\s*(.*)", text or "")
    if not m:
        return None, None
    return int(m.group(1)), m.group(2).strip().lower()


def load_records(fn: Path):
    """Yield (pred_score, pred_level, gold_score, gold_level)."""
    with fn.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f, 1):
            if not line.strip():
                continue
            obj = json.loads(line)
            ps, pl = extract_score_level(obj.get("response", ""))
            gs, gl = extract_score_level(obj.get("labels", ""))
            if None in (ps, pl, gs, gl):
                print(f"Warning: skipped malformed line {idx}")
                continue
            yield ps, pl, gs, gl


def compute_metrics(records):
    preds, plvls, golds, glvls = zip(*records)

    preds = np.array(preds, dtype=float)
    golds = np.array(golds, dtype=float)

    lvl_acc = np.mean([p == g for p, g in zip(plvls, glvls)])
    score_acc = np.mean(preds == golds)
    rmse = math.sqrt(np.mean((preds - golds) ** 2))

    # Handle constant vectors: scipy returns PearsonrResult or raises
    try:
        pearson = pearsonr(preds, golds).statistic
    except Exception:
        pearson = float("nan")

    try:
        spearman = spearmanr(preds, golds).statistic
    except Exception:
        spearman = float("nan")

    return {
        "Leveling accuracy": lvl_acc,
        "Scoring accuracy": score_acc,
        "RMSE": rmse,
        "Pearson r": pearson,
        "Spearman ρ": spearman,
        "Samples": len(preds),
    }


def main():
    ap = argparse.ArgumentParser(
        description="Evaluate pronunciation-scoring predictions"
    )
    ap.add_argument("jsonl", type=Path, help="Path to accuracy.json (JSON-lines)")
    args = ap.parse_args()

    recs = list(load_records(args.jsonl))
    if not recs:
        print("No valid records found.")
        return

    metrics = compute_metrics(recs)
    for k, v in metrics.items():
        print(f"{k:>20}: {v:.6f}" if isinstance(v, float) else f"{k:>20}: {v}")


if __name__ == "__main__":
    main()