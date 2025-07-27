#!/usr/bin/env python3
"""
Create k-fold (default 5) ms-swift JSONL datasets from a CSV.

*Audio paths are not stored in the CSV.*  
Each row has a patient identifier (e.g. `202201201523`).  
Each WAV is now named `<patient>_<kind>_<pos>.wav`, e.g. `03111523_chr_sit.wav`.
Here *kind* indicates the textual content (`chr`, `sentence`, `pinyin`) and
*pos* is the body position (`sit` for seated, `lie` for supine).  For every
patient and kind the script pairs the seated and supine recordings and
generates a single JSON object.

Thus each CSV row expands into  
    K × len(response_cols)  
JSON objects, where **K** is the number of speech kinds (e.g., `chr`, `sentence`, `pinyin`) for which **both** the seated (`sit`) and supine (`lie`) recordings are found. Each object corresponds to one **speech kind × response column** combination:

    {
      "system":   "...",
      "query":    "<prefix><audio>{wav}</audio><suffix>",
      "response": "<value_from_that_single_column>"
    }

Example
-------
```bash
python prepare_osa_single_task_sia_batch_2.py \
    --csv-file label_new_data.csv \
    --output-dir ms_swift_cv \
    --system "You are a helpful medical ASR assistant." \
    --query-prefixs "请听录音并回答：" \
    --query-suffixs "" \
    --response-cols columnA columnB columnC \
    --response-prefixs "" \
    --audio-dir /data/osa_wavs \
    --patient-col patient
```
• The query now automatically appends BMI, 年龄, 颈围 and 体重 so the model no longer needs to predict them.
"""
from __future__ import annotations

import argparse
import json
import pandas as pd
import random
import re
from pathlib import Path
from typing import List, Sequence

# ────────────────────────── New filename‑pattern constants ──────────────────────────
# WAV names now look like:  <patient>_<kind>_<pos>.wav
#   <kind> ∈ {chr, sentence, pinyin}
#   <pos>  ∈ {sit, lie}
POS_MAP = {
    "sit": "坐位",
    "lie": "仰卧位",
}

# Human‑readable phrases for each speech kind
KIND_PHRASES = {
    "chr": "读汉字",
    "sentence": "读句子",
    "pinyin": "读拼音",
}


# ─────────────────────────────── CLI ────────────────────────────────
def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="CSV to k-fold ms-swift JSONL (one column per object)"
    )
    parser.add_argument("--csv-file", required=True, help="Input CSV path")
    parser.add_argument(
        "--output-dir", default="./data/", help="Directory for all JSONL outputs"
    )
    parser.add_argument(
        "--folds", type=int, default=5, help="Number of CV folds (default 5)"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for shuffling"
    )

    # Prompt pieces
    parser.add_argument(
        "--system", required=True, help='Content for the "system" field'
    )
    parser.add_argument(
        "--query-prefixs", nargs="+", required=True, help="Text before <audio>"
    )
    parser.add_argument(
        "--query-suffixs", nargs="+", required=True, help="Text after </audio>"
    )

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
    parser.add_argument(
        "--audio-dir", required=True, help="Directory containing WAV files"
    )
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

    return parser.parse_args()


# ────────────────────────── Helpers ─────────────────────────────────



def make_kfold_indices(total: int, k: int, rng: random.Random):
    """Yield (train_idx, val_idx) lists for each fold."""
    indices = list(range(total))
    rng.shuffle(indices)
    fold_size = total // k
    for i in range(k):
        val = (
            indices[i * fold_size : (i + 1) * fold_size]
            if i < k - 1
            else indices[i * fold_size :]
        )
        train = [idx for idx in indices if idx not in val]
        yield train, val


# ────────────────── Stratified split by OSA severity ──────────────────
def _severity_label(ahi_val) -> str:
    """Return categorical label based on AHI numeric value."""
    try:
        ahi = float(ahi_val)
    except (TypeError, ValueError):
        return "unknown"
    if ahi < 5:
        return "no_osa"
    elif ahi < 15:
        return "mild"
    elif ahi < 30:
        return "moderate"
    else:
        return "severe"


def make_stratified_kfold_indices(df: pd.DataFrame, k: int, rng: random.Random):
    """
    Yield (train_idx, val_idx) for each fold with class‑balanced splits
    over the OSA severity levels derived from the AHI column.
    """
    # Group indices by severity label
    label_to_idx: dict[str, List[int]] = {}
    for idx, val in enumerate(df["AHI"]):
        label = _severity_label(val)
        label_to_idx.setdefault(label, []).append(idx)

    # Shuffle indices within each class
    for lst in label_to_idx.values():
        rng.shuffle(lst)

    # Pre‑allocate empty folds
    folds = [[] for _ in range(k)]

    # Round‑robin distribute samples of each class across folds
    for label, idx_list in label_to_idx.items():
        for i, sample_idx in enumerate(idx_list):
            folds[i % k].append(sample_idx)

    # Yield train/val indices for each fold
    for i in range(k):
        val_idx = folds[i]
        train_idx = [idx for j, f in enumerate(folds) if j != i for idx in f]
        yield train_idx, val_idx


def build_json_line(
    system: str, q_prefix: str, q_suffix: str, wav_paths: List[str], response_text: str
) -> str:
    """
    Return one JSONL line that can embed one or more <audio>...</audio> tags.
    `wav_paths` must be a list; each element will be wrapped in its own tag.
    """
    wav_tags = " 和 ".join(f"<audio>{p}</audio>" for p in wav_paths)
    query = f"{q_prefix}{wav_tags}{q_suffix}"
    obj = {"system": system, "query": query, "response": response_text}
    return json.dumps(obj, ensure_ascii=False)


#
# ───────────────── Balanced JSONL from existing file ────────────────
def generate_balanced_jsonl_from_train(
    train_path: Path,
    balanced_path: Path,
    rng: random.Random,
) -> None:
    """
    Read *train_path* (JSONL), detect **severe** vs **non‑severe** OSA per patient,
    and write a balanced JSONL where each group contributes an equal number of patients.

    Assumes audio filenames inside <audio> tags end with "{patient}_{idx}.wav".
    """
    # Pass 1 ─ collect all lines and map patient → severity
    pat_to_sev: dict[str, str] = {}
    sev_to_pat: dict[str, list[str]] = {"non_severe": [], "severe": []}
    cache: list[tuple[str, str, str]] = []  # (patient, severity, line)

    audio_re = re.compile(r"<audio>[^<]*/([^_/]+)_\d+\.wav</audio>")

    with train_path.open("r", encoding="utf-8") as fin:
        for raw in fin:
            try:
                obj = json.loads(raw)
            except json.JSONDecodeError:
                continue  # skip malformed
            query = obj.get("query", "")
            resp = obj.get("response", "")

            m = audio_re.search(query)
            if not m:
                continue  # no patient id
            pid = m.group(1)

            sev = None
            num_match = re.search(r"[-+]?\d+(?:\.\d+)?", resp)
            if num_match:
                try:
                    ahi_val = float(num_match.group())
                    sev = "severe" if ahi_val >= 30 else "non_severe"
                except ValueError:
                    pass

            cache.append((pid, sev, raw))
            if sev and pid not in pat_to_sev:
                pat_to_sev[pid] = sev
                sev_to_pat[sev].append(pid)

    # Ensure both severe and non-severe present
    if not (sev_to_pat["severe"] and sev_to_pat["non_severe"]):
        print(
            f"Warning: cannot create balanced file from {train_path.name} – both severe and non‑severe classes required."
        )
        return

    min_cnt = min(len(v) for v in sev_to_pat.values())
    selected_patients = set()
    for lev, plist in sev_to_pat.items():
        selected_patients.update(rng.sample(plist, min_cnt))

    # Pass 2 ─ write lines whose patient id is selected
    with balanced_path.open("w", encoding="utf-8") as fout:
        kept = 0
        for pid, sev, raw in cache:
            if pid in selected_patients:
                fout.write(raw)
                kept += 1

    print(
        f"  {balanced_path.name} written with {kept} lines ({min_cnt} patients × 2 levels)."
    )


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
):
    """
    Write JSONL lines for the given indices into *path*.
    A line is produced for every (audio_index x response_col) combination.
    Missing WAV files are skipped silently.
    """
    audio_root = Path(audio_dir)

    with path.open("w", encoding="utf-8") as fout:
        for idx in indices:
            row = df.iloc[idx]
            patient_id = str(row[patient_col]).strip()

            # Extract basic patient info for inclusion in the query
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
                # Format BMI with two decimal places when possible
                try:
                    bmi_float = float(bmi_val)
                    bmi_val_fmt = f"{bmi_float:.1f}"
                except ValueError:
                    bmi_val_fmt = bmi_val  # keep original string if conversion fails
                info_parts.append(f"BMI {bmi_val_fmt}")
            if _valid(neck_val):
                info_parts.append(f"颈围 {neck_val} cm")

            patient_info = (
                "这位患者" if not info_parts else "这是一位" + "，".join(info_parts) + " 的患者"
            )

            # Iterate over each speech kind and build (sit, lie) pairs
            for kind, kind_phrase in KIND_PHRASES.items():
                wav_sit = audio_root / f"{patient_id}_{kind}_sit.wav"
                wav_lie = audio_root / f"{patient_id}_{kind}_lie.wav"

                # Continue only if both recordings exist
                if not (wav_sit.exists() and wav_lie.exists()):
                    continue

                combined_phrase = f"{POS_MAP['sit']}{kind_phrase} 和 {POS_MAP['lie']}{kind_phrase}"

                for col_idx, col in enumerate(response_cols):
                    if col not in row:
                        continue
                    col_val = str(row[col]).strip()
                    if col == "AHI":
                        try:
                            ahi = float(col_val)
                            ahi_str = f"{ahi:.1f}"
                            col_val = f"患者的 AHI 是 {ahi_str}。"
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
                            [str(wav_sit), str(wav_lie)],
                            response_text,
                        )
                        + "\n"
                    )


# ───────────────────────────── Main ────────────────────────────────
def main() -> None:
    args = get_args()
    df = pd.read_csv(args.csv_file)
    n_rows = len(df)
    num_cols = len(args.response_cols)

    # Ensure list‑type arguments line up with response_cols
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
        raise ValueError(
            "--query-prefixs and --query-suffixs must have the same length"
        )

    if n_rows == 0:
        raise RuntimeError("CSV is empty.")

    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)

    for fold_no, (train_idx, val_idx) in enumerate(
        make_stratified_kfold_indices(df, args.folds, rng), 1
    ):
        fold_tag = f"fold{fold_no}"
        train_file = out_root / f"{fold_tag}_train.jsonl"
        val_file = out_root / f"{fold_tag}_val.jsonl"

        write_split(
            df,
            train_idx,
            args.response_cols,
            train_file,
            args.system,
            args.query_prefixs,
            args.query_suffixs,
            args.response_prefixs,
            args.audio_dir,
            args.patient_col,
        )

        write_split(
            df,
            val_idx,
            args.response_cols,
            val_file,
            args.system,
            args.query_prefixs,
            args.query_suffixs,
            args.response_prefixs,
            args.audio_dir,
            args.patient_col,
        )

        # Create balanced train JSONL from the freshly written train_file
        generate_balanced_jsonl_from_train(
            train_file,
            out_root / f"{fold_tag}_train_balanced.jsonl",
            rng,
        )
        # Also create balanced val JSONL
        generate_balanced_jsonl_from_train(
            val_file,
            out_root / f"{fold_tag}_val_balanced.jsonl",
            rng,
        )

        print(f"Done: {fold_tag}")


if __name__ == "__main__":
    main()
