import argparse
import collections
import csv
import json
import os
import re
from pathlib import Path
from tqdm import tqdm


def read_csv_files(file_paths):
    """
    Read multiple CSV files and return their contents.

    Args:
        file_paths (list[str] | list[pathlib.Path]): Paths to CSV files.

    Returns:
        dict[str, list[dict]]: Mapping from file path (as str) to a list of
        rows, where each row is a dict keyed by the CSV header.
    """
    data = {}
    for fp in file_paths:
        path = Path(fp)
        with path.open(newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            data[str(path)] = list(reader)
    return data

# ---------------------------------------------------------------------------

def parse_csv_utterances(csv_path: Path):
    """
    Parse the utterance‑list CSV.

    Expected row format:
        <speaker_id>,<ignored>,<transcript>,[optional…]

    The transcript is taken **only** from the third column (index 2). Rows
    lacking a non‑empty third column are ignored.

    Parameters
    ----------
    csv_path : pathlib.Path | str

    Returns
    -------
    list[dict]
        [
            {"csv_path": "<file>", "session": "<id>", "transcript": "<text>"},
            ...
        ]
    """
    utterances = []
    with Path(csv_path).open(encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row or len(row) < 2:
                continue
            # Column 0 is both the session ID and speaker ID
            spk = session = row[0].strip()

            # Column 2 holds the transcript; skip rows where it is missing/blank
            # if len(row) <= 2 or not row[2].strip():
                # continue
            raw_transcript = row[1].strip()
            mistake = row[2].strip()

            if "\\u" in spk:
                try:
                    spk = bytes(spk, "utf-8").decode("unicode_escape")
                    session = spk
                except UnicodeDecodeError:
                    spk = spk  # fall back
                    session = spk
            else:
                spk = spk
                session = spk

            if "\\u" in raw_transcript:
                try:
                    transcript = bytes(raw_transcript, "utf-8").decode("unicode_escape")
                except UnicodeDecodeError:
                    transcript = raw_transcript  # fall back
            else:
                transcript = raw_transcript
            
            if "\\u" in mistake:
                try:
                    mistake = bytes(mistake, "utf-8").decode("unicode_escape")
                except UnicodeDecodeError:
                    mistake = mistake  # fall back
            else:
                mistake = mistake
            
            utterances.append(
                {
                    "csv_path": str(csv_path),
                    "session": session,
                    "transcript": transcript,
                    "mistake": mistake,
                    "spk": spk,
                }
            )
    return utterances

import itertools  # add near the other imports

# ---------------------------------------------------------------------------
# Transcript‑based matching helpers


_COUNTING_RE = re.compile(
    r"^(一二三四五六七八九十"
    r"(十一十二十三十四十五十六十七十八十九二十)?"
    r")$"
)

def _norm(txt: str) -> str:
    """Remove whitespace & punctuation to normalise transcripts."""
    return re.sub(r"\s+", "", txt)

_NUM_PLACEHOLDERS = {"1-10", "1‑10", "1到10", "一到十", "从一数到十"}

def _is_numeric_placeholder(token: str) -> bool:
    return token in _NUM_PLACEHOLDERS

def _matches_counting(text: str) -> bool:
    return bool(_COUNTING_RE.fullmatch(_norm(text)))

def match_utterances(csv_utts, kaldi_data):
    """
    Pair each row from the CSV with at most one utterance from the Kaldi data
    that has the **same speaker** and **identical transcript** (ignoring spaces).

    Speaker ID
    ----------
    * Prefer the ``spk`` field from `kaldi_data[utt]` (from ``utt2spk``).
    * If absent, derive it with :func:`_speaker_from_utt`.

    Transcript comparison
    ---------------------
    The comparison is performed after calling :func:`_norm` to strip spaces and
    other insignificant whitespace characters.  The CSV transcript can contain
    multiple alternatives separated by '/', and the first alternative that
    matches will be used.

    Special placeholder
    -------------------
    If the CSV transcript is exactly one of the values in ``_NUM_PLACEHOLDERS``
    (e.g. "1-10"), it will match a Kaldi transcript that is a counting string
    recognised by :func:`_matches_counting`.

    Parameters
    ----------
    csv_utts : list[dict]
        Parsed CSV rows, each with keys: ``session``, ``spk``, ``transcript``.
    kaldi_data : dict
        Mapping ``utt_id`` → ``{"wav": ..., "text": ..., "spk": ...}``.

    Returns
    -------
    dict[str, str | None]
        ``{ "<csv_session>_<row#>": "<kaldi_utt_id>" | None }``
    """
    # ------------------------------------------------------------------
    # Build per‑speaker index: speaker → { norm_text → [utt_id …] }
    speaker_text_index = collections.defaultdict(lambda: collections.defaultdict(list))
    counting_index     = collections.defaultdict(list)  # speaker → [utt_id …]

    for utt_id, info in kaldi_data.items():
        spk = info.get("spk")
        # spk = spk.split("_")[0]
        norm_text = _norm(info["text"])
        speaker_text_index[spk][norm_text].append(utt_id)
        if _matches_counting(norm_text):
            counting_index[spk].append(utt_id)

    # ------------------------------------------------------------------
    def _filter_by_speaker(candidates, spk):
        """
        Return a list of utt‑ids from *candidates* whose speaker equals *spk*.
        The check uses kaldi_data[utt]['spk'] if present, otherwise the speaker
        is parsed from the utterance ID via _speaker_from_utt().
        """
        good = []
        for u in candidates:
            real_spk = kaldi_data[u].get("spk")
            # real_spk = real_spk.split("_")[0]
            if real_spk == spk:
                good.append(u)
        return good

    # ------------------------------------------------------------------
    mapping = {}
    for idx, row in enumerate(csv_utts):
        key        = f"{row['session']}_{idx}"
        spk        = row.get("spk") + "_PAR"
        raw_text   = row["transcript"]
        found      = None

        # Try each alternative separated by '/'
        for alt in [part.strip() for part in raw_text.split("/") if part.strip()]:
            # Placeholder "1-10" etc.
            if _is_numeric_placeholder(alt):
                pool = _filter_by_speaker(counting_index.get(spk, []), spk)
                found = pool[0] if pool else None
            else:
                norm_alt = _norm(alt)
                pool = _filter_by_speaker(
                    speaker_text_index.get(spk, {}).get(norm_alt, []), spk
                )
                found = pool[0] if pool else None
            if found:
                break

        mapping[key] = found

    return mapping


def read_kaldi_dir(kaldi_dir):
    """
    Load a Kaldi-style data directory and build a mapping from utterance ID
    to waveform path and transcript (plus speaker ID if available).

    The directory must contain at least:
        - wav.scp   (utterance_id  wav_path)
        - text      (utterance_id  transcript ...)
    Optionally it may include:
        - utt2spk   (utterance_id  speaker_id)

    Args:
        kaldi_dir (str | pathlib.Path): Path to the Kaldi data directory.

    Returns:
        dict[str, dict]: {
            utterance_id: {
                "wav": "/abs/or/rel/path.wav",
                "text": "transcript string",
                "spk": "speaker_id"   # only if utt2spk is present
            }, ...
        }
    """
    kaldi_dir = Path(kaldi_dir)

    wav_scp = kaldi_dir / "wav.scp"
    text_file = kaldi_dir / "text"
    utt2spk_file = kaldi_dir / "utt2spk"

    if not wav_scp.is_file() or not text_file.is_file():
        raise FileNotFoundError(
            f"Expecting wav.scp and text inside {kaldi_dir}, "
            "but one or both are missing."
        )

    # wav.scp: utt -> wav_path
    wav_map = {}
    with wav_scp.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            utt, path = line.split(maxsplit=1)
            wav_map[utt] = path

    # text: utt -> transcript
    text_map = {}
    with text_file.open(encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            pieces = line.split(maxsplit=1)
            if len(pieces) != 2:
                continue  # malformed line
            utt, transcript = pieces
            text_map[utt] = transcript

    # utt2spk (optional)
    spk_map = {}
    if utt2spk_file.is_file():
        with utt2spk_file.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                utt, spk = line.split(maxsplit=1)
                spk = str(int(spk.split("_")[0])) + "_" + spk.split("_")[1]
                spk_map[utt] = spk

    # Merge
    combined = {}
    for utt in itertools.chain(wav_map.keys(), text_map.keys()):
        if utt in wav_map and utt in text_map:
            combined[utt] = {"wav": wav_map[utt], "text": text_map[utt]}
            if spk_map:
                combined[utt]["spk"] = spk_map.get(utt)

    return combined

# ---------------------------------------------------------------------------
def write_kaldi_subset(out_dir: Path, kaldi_data: dict, utt_ids):
    """
    Write a Kaldi‑format data directory at *out_dir* containing only *utt_ids*.

    Files created:
        • wav.scp
        • text
        • utt2spk      (only if speaker information is available)

    Parameters
    ----------
    out_dir : pathlib.Path
    kaldi_data : dict   # as returned by read_kaldi_dir(...)
    utt_ids : Iterable[str]
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    has_spk = any("spk" in kaldi_data[u] for u in utt_ids)

    # wav.scp + text
    with (out_dir / "wav.scp").open("w", encoding="utf-8") as wav_f, \
         (out_dir / "text").open("w", encoding="utf-8") as txt_f, \
         ((out_dir / "utt2spk").open("w", encoding="utf-8") if has_spk else open(os.devnull, "w")) as u2s_f:

        for utt in utt_ids:
            info = kaldi_data[utt]
            wav_f.write(f"{utt} {info['wav']}\n")
            txt_f.write(f"{utt} {info['text']}\n")
            if has_spk:
                spk = info.get("spk", "unknown")
                u2s_f.write(f"{utt} {spk}\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Prepare PUMCH data: load one or more CSV metadata files."
    )
    parser.add_argument(
        "csv_files",
        nargs="*",
        help="Zero or more CSV metadata files to load."
    )
    parser.add_argument(
        "--kaldi-dir",
        type=str,
        default=None,
        help="Optional path to a Kaldi-format data directory (wav.scp/text[/utt2spk])."
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="If provided, write the matched subset to this Kaldi data directory."
    )
    parser.add_argument(
        "--align-json",
        type=str,
        default=None,
        help="If provided, write the alignment mapping to this JSON file."
    )
    args = parser.parse_args()

    csv_data = read_csv_files(args.csv_files)
    # Example action: print a summary of each file
    for path, rows in csv_data.items():
        print(f"{path}: {len(rows)} rows loaded")

    # If a Kaldi data directory is provided, load it as well.
    if args.kaldi_dir:
        kaldi_data = read_kaldi_dir(args.kaldi_dir)
        print(f"{args.kaldi_dir}: {len(kaldi_data)} utterances loaded")
        # print(kaldi_data)  # (Optional) Remove or comment out to avoid flooding stdout

    # -------------------------------------------------------------------
    # If both CSV files and Kaldi data are present, attempt alignment.
    if args.kaldi_dir and args.csv_files:
        all_csv_utts = []
        for csv_file in args.csv_files:
            all_csv_utts.extend(parse_csv_utterances(csv_file))

        align_map = match_utterances(all_csv_utts, kaldi_data)
        matched = sum(1 for v in align_map.values() if v)
        print(f"\nMatched {matched}/{len(align_map)} CSV rows to Kaldi utterances.")

        # Show first ten results as a sanity check
        print("  preview:")
        for k, v in list(align_map.items())[:10]:
            print(f"    {k}  ->  {v}")

        # Dump alignment details to JSON if requested
        if args.align_json:
            # Build a rich JSON list with per‑row details
            json_rows = []
            for idx, row in enumerate(all_csv_utts):
                key = f"{row['session']}_{idx}"
                kaldi_id = align_map[key]
                entry = {
                    "key": key,                       # CSV row key
                    "csv_speaker_id": row.get("spk"),
                    "csv_transcript": row["transcript"],
                    "csv_mistake": row["mistake"],
                    "kaldi_utt_id": kaldi_id,
                    "kaldi_speaker_id": None,
                    "kaldi_transcript": None,
                    "wav_path": None,
                }
                if kaldi_id:
                    info = kaldi_data[kaldi_id]
                    kaldi_spk = info.get("spk")
                    entry.update(
                        {
                            "kaldi_speaker_id": kaldi_spk,
                            "kaldi_transcript": info["text"],
                            "wav_path": info["wav"],
                        }
                    )
                json_rows.append(entry)

            with open(args.align_json, "w", encoding="utf-8") as jf:
                json.dump(json_rows, jf, ensure_ascii=False, indent=2)

            print(f"Alignment details saved to {args.align_json}")

        # ----------------------------------------------------------------
        # Write subset if requested
        if args.out_dir:
            matched_utts = [u for u in align_map.values() if u]
            write_kaldi_subset(Path(args.out_dir), kaldi_data, matched_utts)
            print(f"Wrote {len(matched_utts)} utterances to {args.out_dir}")