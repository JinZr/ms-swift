#!/usr/bin/env python3
"""
osa_eval_binary_only.py — Evaluate binary OSA severity predictions (<SEVERE> / <NON_SEVERE>).

Usage:
    python osa_eval_binary_only.py <infer_results_dir> <output_csv>
"""

import csv
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

try:  # optional, nicer F1 if available
    from sklearn.metrics import f1_score

    SKLEARN_OK = True
except ImportError:
    SKLEARN_OK = False


# ---------- Patterns ----------
# 捕获 “重度 OSA / 中度 OSA / 轻度 OSA / OSA 阴性 / 非重度 OSA” 等表述
LEVEL_RE = re.compile(r"(?:重度|中度|轻度)\s*OSA|OSA\s*阴性|非重度\s*OSA", re.IGNORECASE)
TAG_RE = re.compile(r"<\s*(SEVERE|NON_SEVERE)\s*>", re.IGNORECASE)

# ---------- Helpers ----------
def load_entries(path: Path):
    txt = path.read_text(encoding="utf-8").strip()
    try:  # 尝试整体 JSON
        obj = json.loads(txt)
        return obj if isinstance(obj, list) else [obj]
    except json.JSONDecodeError:
        # 行分隔 ND-JSON
        return [json.loads(line) for line in txt.splitlines() if line.strip()]


def extract_level(text: str):
    m = LEVEL_RE.search(text or "")
    return m.group(0).strip().replace(" ", "").upper() if m else None


def extract_tag(text: str):
    m = TAG_RE.search(text or "")
    return m.group(1).upper() if m else None


def macro_f1(y_true, y_pred, labels):
    """Fallback F1 实现（宏平均）。"""
    tp = Counter()
    fp = Counter()
    fn = Counter()
    for t, p in zip(y_true, y_pred):
        if t == p:
            tp[t] += 1
        else:
            fp[p] += 1
            fn[t] += 1

    f1s = {}
    for lbl in labels:
        precision = tp[lbl] / (tp[lbl] + fp[lbl]) if (tp[lbl] + fp[lbl]) else 0.0
        recall = tp[lbl] / (tp[lbl] + fn[lbl]) if (tp[lbl] + fn[lbl]) else 0.0
        if precision + recall:
            f1s[lbl] = 2 * precision * recall / (precision + recall)
        else:
            f1s[lbl] = 0.0
    macro = sum(f1s.values()) / len(labels) if labels else 0.0
    return f1s, macro


# ---------- Main Evaluation ----------
def evaluate(entries):
    y_true_lbl, y_pred_lbl = [], []

    for obj in entries:
        pred = obj.get("response", "")
        true = obj.get("labels", "")

        y_pred_lbl.append(extract_tag(pred))
        y_true_lbl.append(extract_tag(true))

    # Drop examples where either side is missing a tag
    valid_mask = [t is not None and p is not None
                  for t, p in zip(y_true_lbl, y_pred_lbl)]
    y_true_eval = [t for t, m in zip(y_true_lbl, valid_mask) if m]
    y_pred_eval = [p for p, m in zip(y_pred_lbl, valid_mask) if m]

    # Confusion matrix
    tp = sum(t == "SEVERE" and p == "SEVERE" for t, p in zip(y_true_eval, y_pred_eval))
    tn = sum(t == "NON_SEVERE" and p == "NON_SEVERE" for t, p in zip(y_true_eval, y_pred_eval))
    fp = sum(t == "NON_SEVERE" and p == "SEVERE" for t, p in zip(y_true_eval, y_pred_eval))
    fn = sum(t == "SEVERE" and p == "NON_SEVERE" for t, p in zip(y_true_eval, y_pred_eval))

    def safe_div(num, denom):
        return num / denom if denom else float("nan")

    accuracy     = safe_div(tp + tn, tp + tn + fp + fn)
    sensitivity  = safe_div(tp, tp + fn)   # Recall for SEVERE
    specificity  = safe_div(tn, tn + fp)   # True‑negative rate
    precision    = safe_div(tp, tp + fp)   # Positive predictive value

    # Macro‑averaged F1 across the two classes
    labels_bin = ["SEVERE", "NON_SEVERE"]
    if SKLEARN_OK:
        f1_macro = f1_score(y_true_eval, y_pred_eval, labels=labels_bin, average="macro")
        f1_percls = dict(
            zip(labels_bin, f1_score(y_true_eval, y_pred_eval, labels=labels_bin, average=None))
        )
    else:
        f1_percls, f1_macro = macro_f1(y_true_eval, y_pred_eval, labels_bin)

    return accuracy, sensitivity, specificity, precision, f1_macro, f1_percls


def main():
    usage_msg = (
        "Usage:\n"
        "  python osa_eval_binary_only.py <infer_results_dir> <output_csv>\n"
        "\n"
        "  <infer_results_dir>  Directory that contains one or more *.jsonl files\n"
        "                       (e.g. exp_dir/checkpoint-*/infer_result/20240625*.jsonl)\n"
        "  <output_csv>         Path to write the aggregated metrics as CSV\n"
    )

    # --- Parse CLI ---
    if len(sys.argv) not in {2, 3}:
        print(usage_msg)
        sys.exit(1)

    infer_dir = Path(sys.argv[1])
    if not infer_dir.exists() or not infer_dir.is_dir():
        print(f"Directory not found: {infer_dir}")
        sys.exit(1)

    # Default output CSV next to directory if not given
    if len(sys.argv) == 3:
        csv_path = Path(sys.argv[2])
    else:
        csv_path = infer_dir / "osa_eval_results.csv"

    # --- Gather *.jsonl files ---
    jsonl_files = sorted(infer_dir.glob("*/infer_result/*.jsonl"))
    if not jsonl_files:
        print(f"No .jsonl files found under {infer_dir}")
        sys.exit(1)

    # --- Evaluate each file ---
    rows = []
    for fp in jsonl_files:
        try:
            entries = load_entries(fp)
            acc, sens, spec, prec, f1_macro, f1_percls = evaluate(entries)
            rows.append(
                {
                    "file": fp.name,
                    "accuracy": f"{acc:.4f}",
                    "sensitivity": f"{sens:.4f}",
                    "specificity": f"{spec:.4f}",
                    "precision": f"{prec:.4f}",
                    "f1_macro": f"{f1_macro:.4f}",
                }
            )
            print(
                f"[OK] {fp.name}: "
                f"acc={acc:.2%}, sens={sens:.2%}, spec={spec:.2%}, prec={prec:.2%}, "
                f"F1={f1_macro:.3f}"
            )
        except Exception as exc:
            print(f"[ERROR] {fp.name}: {exc}")

    # --- Write CSV ---
    fieldnames = ["file", "accuracy", "sensitivity", "specificity",
                  "precision", "f1_macro"]
    with csv_path.open("w", newline="") as fout:
        writer = csv.DictWriter(fout, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nWrote aggregated results to {csv_path.resolve()}")


if __name__ == "__main__":
    main()
