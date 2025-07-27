#!/usr/bin/env python3
"""
Create k-fold (default 5) ms-swift JSONL datasets from a CSV.

*Audio paths are not stored in the CSV.*  
Each row has a patient identifier (e.g. `202201201523`).  
For every requested audio index (default 1-6) the script looks for
`<audio_dir>/<patient>_<idx>.wav` and generates a separate JSON object.

Thus each CSV row expands into  
    len(audio_indices) x len(response_cols)  
JSON objects, one per **audio index x response column**:

    {
      "system":   "...",
      "query":    "<prefix><audio>{wav}</audio><suffix>",
      "response": "<value_from_that_single_column>"
    }

Example
-------
```bash
python prepare_osa_single_task.py \
    --csv-file label_new_data.csv \
    --output-dir ms_swift_cv \
    --system "You are a helpful medical ASR assistant." \
    --query-prefixs "请听录音并回答：" \
    --query-suffixs "" \
    --response-cols columnA columnB columnC \
    --response-prefixs "" \
    --audio-dir /data/osa_wavs \
    --patient-col patient \
    --audio-indices 1 2 3 4 5 6
```
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import List, Sequence

import pandas as pd


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
AUDIO_INDEX_PHRASES = {
    1: "坐位读拼音",
    2: "坐位读汉字",
    3: "坐位读句子",
    4: "仰卧位读拼音",
    5: "仰卧位读汉字",
    6: "仰卧位读句子",
}


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


def build_json_line(
    system: str, q_prefix: str, q_suffix: str, wav_path: str, response_text: str
) -> str:
    """Return one JSONL line for ms-swift."""
    query = f"{q_prefix}<audio>{wav_path}</audio>{q_suffix}"
    obj = {"system": system, "query": query, "response": response_text}
    return json.dumps(obj, ensure_ascii=False)


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

            for ai in audio_indices:
                wav_path = audio_root / f"{patient_id}_{ai}.wav"
                if not wav_path.exists():
                    continue

                # 获取音频标签和口语化短语
                ai_label = AUDIO_INDEX_LABELS.get(ai, f"音频{ai}")
                ai_phrase = AUDIO_INDEX_PHRASES.get(ai, ai_label)

                for col_idx, col in enumerate(response_cols):
                    if col not in row:
                        continue
                    response_text = (
                        f"{response_prefixs[col_idx]}{str(row[col]).strip()}"
                    )
                    raw_q = query_prefixs[col_idx]

                    # 去掉“请听录音并”前缀（若存在），得到尾部描述
                    tail = raw_q
                    if tail.startswith("请听录音并"):
                        tail = tail[len("请听录音并") :]
                    # 构造自然语言查询前缀
                    natural_prefix = f"请听说话人{ai_phrase}的录音，并{tail}"

                    fout.write(
                        build_json_line(
                            system,
                            natural_prefix,
                            query_suffixs[col_idx],
                            str(wav_path),
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
        make_kfold_indices(n_rows, args.folds, rng), 1
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
            args.audio_indices,
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
            args.audio_indices,
        )

        print(
            f"Done: {fold_tag}: {len(train_idx)*len(args.response_cols)*len(args.audio_indices)} train + "
            f"{len(val_idx)*len(args.response_cols)*len(args.audio_indices)} val lines"
        )


if __name__ == "__main__":
    main()
