#!/usr/bin/env python3
"""
osa_eval.py  —  Evaluate OSA level predictions & AHI correlation.

Usage:
    python osa_eval.py path/to/file.json
"""

import csv
import json
import math
import re
import statistics
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

# 捕获 “…AHI 是 23.4” 或 “AHI=23.4” 等

AHI_RE = re.compile(r"AHI(?:\s*[:：=]\s*|\s*是\s*)([0-9]+(?:\.[0-9]+)?)", re.IGNORECASE)

# 捕获说话人 ID（如 202202221526）——假设出现在 <audio> 标签文件名中
SPEAKER_RE = re.compile(r"/([0-9]{8,})_[^/<>]*\.wav", re.IGNORECASE)

# ---------- Level Priority & Voting ----------
# Ordinal severity ranking used for deterministic, risk‑averse tie‑breaking.
# Keys correspond to the uppercase, whitespace‑stripped strings returned by
# `extract_level()`.
LEVEL_PRIORITY = {
    "OSA阴性": 0,     # Negative / no OSA
    "非重度OSA": 0,   # “Not severe OSA” – treat the same as negative
    "轻度OSA": 1,
    "中度OSA": 2,
    "重度OSA": 3,
}

def majority_vote(levels):
    """
    Return the majority‑voted OSA level.
    ‑ If one label clearly has more votes, return it.
    ‑ When there is a tie (e.g. 1:1), break the tie by choosing the level with
      the **higher severity rank** (LEVEL_PRIORITY).  This is a conservative
      clinical assumption that errs on the side of caution.
    """
    # Filter out None values
    levels = [lvl for lvl in levels if lvl is not None]
    if not levels:
        return None

    counts = Counter(levels)
    max_cnt = max(counts.values())
    # Collect all levels that reach the top vote count
    top_levels = [lvl for lvl, cnt in counts.items() if cnt == max_cnt]

    if len(top_levels) == 1:
        return top_levels[0]

    # Tie‑break: pick the level with higher severity rank
    return max(top_levels, key=lambda l: LEVEL_PRIORITY.get(l, -1))

def severity_label(ahi: float):
    """
    Binary severity label based on AHI threshold.

    Parameters
    ----------
    ahi : float
        Apnea–Hypopnea Index.

    Returns
    -------
    str | None
        "SEVERE"  if ahi > 30
        "NON_SEVERE" if ahi <= 30
        None when ahi is NaN
    """
    if ahi is None or math.isnan(ahi):
        return None
    return "SEVERE" if ahi > 30 else "NON_SEVERE"

def extract_speaker_id(obj):
    """从 messages[*].content 中抽取说话人 ID; 若未找到则返回 None."""
    for msg in obj.get("messages", []):
        m = SPEAKER_RE.search(msg.get("content", ""))
        if m:
            return m.group(1)
    return None

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


def extract_ahi(text: str):
    m = AHI_RE.search(text or "")
    return float(m.group(1)) if m else math.nan


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


def pearson_r(xs, ys):
    mean_x = statistics.mean(xs)
    mean_y = statistics.mean(ys)
    num = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    den = math.sqrt(
        sum((x - mean_x) ** 2 for x in xs) * sum((y - mean_y) ** 2 for y in ys)
    )
    return num / den if den else float("nan")



# ---------- Main Evaluation ----------
def evaluate(entries):
    """
    在说话人 (subject) 层面计算指标：
    - OSA level 使用多数投票
    - AHI 使用均值
    """
    by_spk = defaultdict(lambda: {
        "pred_lvl": [],
        "true_lvl": [],
        "pred_ahi": [],
        "true_ahi": [],
    })

    # ----- 聚合每条记录 -----
    for obj in entries:
        spk_id = extract_speaker_id(obj) or f"UNK_{id(obj)}"

        pred_txt = obj.get("response", "")
        true_txt = obj.get("labels", "")

        by_spk[spk_id]["pred_lvl"].append(extract_level(pred_txt))
        by_spk[spk_id]["true_lvl"].append(extract_level(true_txt))
        by_spk[spk_id]["pred_ahi"].append(extract_ahi(pred_txt))
        by_spk[spk_id]["true_ahi"].append(extract_ahi(true_txt))

    # ----- 说话人层面投票 / 均值 -----
    y_true_lvl, y_pred_lvl = [], []
    y_true_ahi, y_pred_ahi = [], []

    for spk_id, vals in by_spk.items():
        # --- Level (多数投票) ---
        pred_lvl = majority_vote(vals["pred_lvl"])
        true_lvl = majority_vote(vals["true_lvl"])
        y_pred_lvl.append(pred_lvl)
        y_true_lvl.append(true_lvl)

        # --- AHI (均值) ---
        preds_ahi = [v for v in vals["pred_ahi"] if not math.isnan(v)]
        trues_ahi = [v for v in vals["true_ahi"] if not math.isnan(v)]
        y_pred_ahi.append(sum(preds_ahi) / len(preds_ahi) if preds_ahi else math.nan)
        y_true_ahi.append(sum(trues_ahi) / len(trues_ahi) if trues_ahi else math.nan)

    # ----- AHI‑based Severity Metrics -----
    y_true_sev = [severity_label(a) for a in y_true_ahi]
    y_pred_sev = [severity_label(a) for a in y_pred_ahi]

    valid_mask = [t is not None and p is not None for t, p in zip(y_true_sev, y_pred_sev)]
    y_true_eval = [t for t, m in zip(y_true_sev, valid_mask) if m]
    y_pred_eval = [p for p, m in zip(y_pred_sev, valid_mask) if m]

    # Confusion‑matrix counts
    tp = sum(t == "SEVERE" and p == "SEVERE" for t, p in zip(y_true_eval, y_pred_eval))
    tn = sum(t == "NON_SEVERE" and p == "NON_SEVERE" for t, p in zip(y_true_eval, y_pred_eval))
    fp = sum(t == "NON_SEVERE" and p == "SEVERE" for t, p in zip(y_true_eval, y_pred_eval))
    fn = sum(t == "SEVERE" and p == "NON_SEVERE" for t, p in zip(y_true_eval, y_pred_eval))

    def safe_div(num, denom):
        return num / denom if denom else float("nan")

    sensitivity = safe_div(tp, tp + fn)      # Recall (SEVERE)
    specificity = safe_div(tn, tn + fp)      # True‑negative rate
    precision   = safe_div(tp, tp + fp)      # Positive predictive value
    accuracy    = safe_div(tp + tn, tp + tn + fp + fn)

    # Macro F1 for completeness
    labels_bin = ["SEVERE", "NON_SEVERE"]
    if SKLEARN_OK:
        f1_macro = f1_score(y_true_eval, y_pred_eval, labels=labels_bin, average="macro")
        f1_percls = dict(
            zip(labels_bin, f1_score(y_true_eval, y_pred_eval, labels=labels_bin, average=None))
        )
    else:
        f1_percls, f1_macro = macro_f1(y_true_eval, y_pred_eval, labels_bin)

    # ----- Pearson r (AHI) -----
    ahi_pairs = [
        (p, t)
        for p, t in zip(y_pred_ahi, y_true_ahi)
        if not math.isnan(p) and not math.isnan(t)
    ]
    if len(ahi_pairs) >= 2:
        preds, trues = zip(*ahi_pairs)
        r = pearson_r(preds, trues)
    else:
        r = float("nan")

    return accuracy, sensitivity, specificity, precision, f1_macro, f1_percls, r


def main():
    usage_msg = (
        "Usage:\n"
        "  python osa_eval.py <infer_results_dir> <output_csv>\n"
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
        csv_path = infer_dir / "osa_eval_results_subject_lvl.csv"

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
            acc, sens, spec, prec, f1_macro, f1_percls, r = evaluate(entries)
            rows.append(
                {
                    "file": fp.name,
                    "accuracy": f"{acc:.4f}",
                    "sensitivity": f"{sens:.4f}",
                    "specificity": f"{spec:.4f}",
                    "precision": f"{prec:.4f}",
                    "f1_macro": f"{f1_macro:.4f}",
                    "pearson_r": f"{r:.4f}",
                }
            )
            print(
                f"[OK] {fp.name}: "
                f"acc={acc:.2%}, sens={sens:.2%}, spec={spec:.2%}, prec={prec:.2%}, "
                f"F1={f1_macro:.3f}, r={r:.4f}"
            )
        except Exception as exc:
            print(f"[ERROR] {fp.name}: {exc}")

    # --- Write CSV ---
    fieldnames = ["file", "accuracy", "sensitivity", "specificity", "precision", "f1_macro", "pearson_r"]
    with csv_path.open("w", newline="") as fout:
        writer = csv.DictWriter(fout, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nWrote aggregated results to {csv_path.resolve()}")


if __name__ == "__main__":
    main()
