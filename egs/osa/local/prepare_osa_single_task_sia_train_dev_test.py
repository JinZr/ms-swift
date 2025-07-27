#!/usr/bin/env python3
"""
Create **train / dev / test** ms-swift JSONL datasets from a CSV.

*Audio paths are not stored in the CSV.*  
Each row has a patient identifier (e.g. `202201201523`).  
For every requested audio index (default 1-6) the script looks for
`<audio_dir>/<patient>_<idx>.wav` and generates a separate JSON object.

Thus each CSV row expands into  
    len(audio_indices) × len(response_cols)  
JSON objects, one per **audio index × response column**:

    {
      "system":   "...",
      "query":    "<prefix><audio>{wav}</audio><suffix>",
      "response": "<value_from_that_single_column>"
    }

New in this version
-------------------
* Replaces the old 5-fold cross-validation logic with a **single train / dev / test** split.
* `--dev-ratio` and `--test-ratio` (default 0.1 each) control the split sizes.  
  The remainder of the data goes to *train*.
* Splits are **stratified by OSA severity** (AHI < 30 vs ≥ 30) so that all three splits
  contain a balanced representation of severe / non-severe subjects.
* For each of the three splits an additional `_*_balanced.jsonl` file is emitted where
  the number of severe and non-severe patients is exactly equal (down-sampled from the
  larger class), mirroring the behaviour of the original script.

Example
-------
```bash
python prepare_osa_single_task_sia_train_dev_test.py \
    --csv-file label_new_data.csv \
    --output-dir ms_swift \
    --system "You are a helpful medical ASR assistant." \
    --query-prefixs "请听录音并回答：" \
    --query-suffixs "" \
    --response-cols columnA columnB columnC \
    --response-prefixs "" \
    --audio-dir /data/osa_wavs \
    --patient-col patient \
    --audio-indices 1 2 3 4 5 6 \
    --dev-ratio 0.12 --test-ratio 0.12
```

The query automatically appends BMI、年龄、颈围 and 体重 so the model no longer needs to predict them.
"""
from __future__ import annotations

import argparse
import json
import pandas as pd
import random
import re
from pathlib import Path
from typing import List, Sequence

# ─────────────────────────────── CLI ────────────────────────────────

def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="CSV → train / dev / test ms-swift JSONL (one column per object)"
    )
    parser.add_argument("--csv-file", required=True, help="Input CSV path")
    parser.add_argument(
        "--output-dir", default="./data/", help="Directory for all JSONL outputs"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for shuffling")

    # Split sizes
    parser.add_argument(
        "--dev-ratio",
        type=float,
        default=0.15,
        help="Fraction of subjects to allocate to the dev set (default 0.1)",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.15,
        help="Fraction of subjects to allocate to the test set (default 0.1)",
    )

    # Prompt pieces
    parser.add_argument("--system", required=True, help='Content for the "system" field')
    parser.add_argument("--query-prefixs", nargs="+", required=True, help="Text before <audio>")
    parser.add_argument("--query-suffixs", nargs="+", required=True, help="Text after </audio>")

    # Column control
    parser.add_argument(
        "--response-cols",
        nargs="+",
        required=True,
        metavar="COL",
        help="CSV columns; every entry becomes its own JSON object",
    )
    parser.add_argument(
        "--response-prefixs",
        nargs="+",
        required=True,
        metavar="PRE",
        help="Text before response-val",
    )
    parser.add_argument("--audio-dir", required=True, help="Directory containing WAV files")
    parser.add_argument(
        "--patient-col",
        default="patient",
        help="CSV column holding the patient id (default: patient)",
    )
    parser.add_argument(
        "--audio-indices",
        type=int,
        nargs="+",
        default=[1, 2, 3, 4, 5, 6],
        metavar="N",
        help="Indices appended to <patient>_ when forming WAV filenames",
    )

    args = parser.parse_args()

    if args.dev_ratio < 0 or args.test_ratio < 0 or args.dev_ratio + args.test_ratio >= 1:
        parser.error("--dev-ratio and --test-ratio must be ≥ 0 and sum to < 1.")

    return args


# ────────────────────────── Helpers ─────────────────────────────────

# 音频索引标签
AUDIO_INDEX_LABELS = {
    1: "坐位读拼音",
    2: "坐位读汉字",
    3: "坐位读句子",
    4: "仰卧位读拼音",
    5: "仰卧位读汉字",
    6: "仰卧位读句子",
}

# 更口语化的描述，用于拼接到查询前缀
AUDIO_INDEX_PHRASES = AUDIO_INDEX_LABELS.copy()

# ────────────────────────── Paired positions ──────────────────────────
# (seated position, supine position) pairs for the SAME textual content
AUDIO_INDEX_PAIRS = [(1, 4), (2, 5), (3, 6)]


# ────────────────── Stratified train / dev / test split ──────────────────

def _severity_label(ahi_val) -> str:
    """Return categorical label based on AHI numeric value."""
    try:
        ahi = float(ahi_val)
    except (TypeError, ValueError):
        return "unknown"
    return "severe" if ahi >= 30 else "non_severe"


def make_train_dev_test_indices(
    df: pd.DataFrame, dev_ratio: float, test_ratio: float, rng: random.Random
) -> tuple[Sequence[int], Sequence[int], Sequence[int]]:
    """Return (train_idx, dev_idx, test_idx) lists with class-stratified sampling."""
    # Group indices by severity
    sev_to_idx: dict[str, List[int]] = {"non_severe": [], "severe": []}
    for idx, val in enumerate(df["AHI"]):
        sev_to_idx[_severity_label(val)].append(idx)

    # Shuffle within each class
    for lst in sev_to_idx.values():
        rng.shuffle(lst)

    dev_idx: list[int] = []
    test_idx: list[int] = []

    # Determine sample counts per class so that dev & test are balanced
    for label, idx_list in sev_to_idx.items():
        n_total = len(idx_list)
        n_test = int(round(n_total * test_ratio))
        n_dev = int(round(n_total * dev_ratio))

        # Edge cases – always keep at least 1 if the class exists
        if 0 < n_total < (n_test + n_dev):
            n_test = min(n_test, n_total)
            n_dev = max(0, n_total - n_test)

        test_idx += idx_list[:n_test]
        dev_idx += idx_list[n_test : n_test + n_dev]
        # Remaining go to train automatically
        sev_to_idx[label] = idx_list[n_test + n_dev :]

    # Combine leftovers for train
    train_idx = sev_to_idx["non_severe"] + sev_to_idx["severe"]

    # Final shuffle to avoid class grouping inside splits
    rng.shuffle(train_idx)
    rng.shuffle(dev_idx)
    rng.shuffle(test_idx)

    return train_idx, dev_idx, test_idx


# ───────────────── Serialization helpers (unchanged) ─────────────────

def build_json_line(
    system: str, q_prefix: str, q_suffix: str, wav_paths: List[str], response_text: str
) -> str:
    wav_tags = " 和 ".join(f"<audio>{p}</audio>" for p in wav_paths)
    query = f"{q_prefix}{wav_tags}{q_suffix}"
    obj = {"system": system, "query": query, "response": response_text}
    return json.dumps(obj, ensure_ascii=False)


# ────────── Generate balanced severe / non-severe JSONL ──────────

def generate_balanced_jsonl_from_split(
    src_path: Path,
    balanced_path: Path,
    rng: random.Random,
) -> None:
    """Down-sample *src_path* so that severe vs non-severe patients are equal."""
    severity_kw = {
        "non_severe": "非重度 OSA",
        "severe": "重度 OSA",
    }

    pat_to_sev: dict[str, str] = {}
    sev_to_pat: dict[str, list[str]] = {k: [] for k in severity_kw}
    cache: list[tuple[str, str, str]] = []  # (patient, severity, raw_line)

    audio_re = re.compile(r"<audio>[^<]*/([^_/]+)_\d+\.wav</audio>")

    with src_path.open("r", encoding="utf-8") as fin:
        for raw in fin:
            try:
                obj = json.loads(raw)
            except json.JSONDecodeError:
                continue
            query = obj.get("query", "")
            resp = obj.get("response", "")

            m = audio_re.search(query)
            if not m:
                continue  # no patient id
            pid = m.group(1)

            sev = None
            for lbl, kw in severity_kw.items():
                if kw in resp:
                    sev = lbl
                    break

            cache.append((pid, sev, raw))
            if sev and pid not in pat_to_sev:
                pat_to_sev[pid] = sev
                sev_to_pat[sev].append(pid)

    if not all(sev_to_pat[lev] for lev in severity_kw):
        print(
            f"Warning: cannot create balanced file from {src_path.name} – both severe and non-severe classes required."
        )
        return

    min_cnt = min(len(v) for v in sev_to_pat.values())
    selected_patients = set()
    for lev, plist in sev_to_pat.items():
        selected_patients.update(rng.sample(plist, min_cnt))

    with balanced_path.open("w", encoding="utf-8") as fout:
        kept = 0
        for pid, _, raw in cache:
            if pid in selected_patients:
                fout.write(raw)
                kept += 1

    print(f"  {balanced_path.name} written with {kept} lines ({min_cnt} patients × 2 levels).")


# ────────── JSONL writer (mostly unchanged except for var names) ──────────

def write_split(
    df: pd.DataFrame,
    indices: Sequence[int],
    response_cols: List[str],
    path: Path,
    system: str,
    query_prefixs: List[str],
    query_suffixs: List[str],
    response_prefixs: List[str],
    audio_dir: str,
    patient_col: str,
    audio_indices: List[int],
):
    audio_root = Path(audio_dir)

    with path.open("w", encoding="utf-8") as fout:
        for idx in indices:
            row = df.iloc[idx]
            patient_id = str(row[patient_col]).strip()

            # Extract basic patient info
            age_val = str(row.get("age", "")).strip()
            bmi_val = str(row.get("BMI", "")).strip()
            neck_val = str(row.get("neck_circum", "")).strip()
            weight_val = str(row.get("weight", "")).strip()

            def _valid(v: str) -> bool:
                return v and v.lower() != "nan"

            info_parts = []
            if _valid(age_val):
                info_parts.append(f"年龄 {age_val} 岁")
            if _valid(weight_val):
                info_parts.append(f"体重 {weight_val} kg")
            if _valid(bmi_val):
                info_parts.append(f"BMI {bmi_val}")
            if _valid(neck_val):
                info_parts.append(f"颈围 {neck_val} cm")

            patient_info = (
                "这位患者" if not info_parts else "这是一位" + "，".join(info_parts) + " 的患者"
            )

            # Iterate over (seated, supine) pairs
            for ai1, ai2 in AUDIO_INDEX_PAIRS:
                if ai1 not in audio_indices or ai2 not in audio_indices:
                    continue

                wav1 = audio_root / f"{patient_id}_{ai1}.wav"
                wav2 = audio_root / f"{patient_id}_{ai2}.wav"
                if not wav1.exists() or not wav2.exists():
                    continue

                phrase1 = AUDIO_INDEX_PHRASES.get(ai1, f"音频{ai1}")
                phrase2 = AUDIO_INDEX_PHRASES.get(ai2, f"音频{ai2}")
                combined_phrase = f"{phrase1} 和 {phrase2}"

                for col_idx, col in enumerate(response_cols):
                    if col not in row:
                        continue
                    col_val = str(row[col]).strip()
                    if col == "AHI":
                        try:
                            ahi = float(col_val)
                            severity = "非重度 OSA" if ahi < 30 else "重度 OSA"
                            col_val = f"{severity}，患者的 AHI 是 {ahi:.1f}。"
                        except ValueError:
                            pass

                    response_text = f"{response_prefixs[col_idx]}{col_val}"

                    raw_q = query_prefixs[col_idx]
                    tail = raw_q[len("请听录音并") :] if raw_q.startswith("请听录音并") else raw_q
                    natural_prefix = (
                        f"{patient_info}, 以下提供两段说话人以{combined_phrase}朗读相同文本的录音，并{tail}"
                    )

                    fout.write(
                        build_json_line(
                            system,
                            natural_prefix,
                            query_suffixs[col_idx],
                            [str(wav1), str(wav2)],
                            response_text,
                        )
                        + "\n"
                    )


# ───────────────────────────── Main ────────────────────────────────

def main() -> None:
    args = get_args()
    df = pd.read_csv(args.csv_file)
    if df.empty:
        raise RuntimeError("CSV is empty.")

    num_cols = len(args.response_cols)

    # Broadcast list-type arguments
    def _broadcast(lst: List[str], name: str) -> List[str]:
        if len(lst) == 1:
            return lst * num_cols
        if len(lst) != num_cols:
            raise ValueError(
                f"--{name} length must be 1 or exactly match --response-cols ({num_cols})"
            )
        return lst

    args.query_prefixs = _broadcast(args.query_prefixs, "query-prefixs")
    args.query_suffixs = _broadcast(args.query_suffixs, "query-suffixs")
    args.response_prefixs = _broadcast(args.response_prefixs, "response-prefixs")

    if len(args.query_prefixs) != len(args.query_suffixs):
        raise ValueError("--query-prefixs and --query-suffixs must have the same length")

    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)

    # ------------------------ split the dataset ------------------------
    train_idx, dev_idx, test_idx = make_train_dev_test_indices(
        df, args.dev_ratio, args.test_ratio, rng
    )

    split_specs = [
        (train_idx, "train"),
        (dev_idx, "dev"),
        (test_idx, "test"),
    ]

    for idxs, tag in split_specs:
        jsonl_path = out_root / f"{tag}.jsonl"

        write_split(
            df,
            idxs,
            args.response_cols,
            jsonl_path,
            args.system,
            args.query_prefixs,
            args.query_suffixs,
            args.response_prefixs,
            args.audio_dir,
            args.patient_col,
            args.audio_indices,
        )

        # Balanced version
        generate_balanced_jsonl_from_split(
            jsonl_path,
            out_root / f"{tag}_balanced.jsonl",
            rng,
        )

        print(f"Done writing {tag} split (patients: {len(set(idxs))}).")


if __name__ == "__main__":
    main()
