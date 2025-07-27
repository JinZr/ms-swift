#!/usr/bin/env python3
# kaldi_to_msswift_jsonl.py
#
# Convert a Kaldi-format data directory (wav.scp, text, segments …)
# into a JSONL file where each line has keys:
#   "system", "query", "response"
#
# Usage example
#   python kaldi_to_msswift_jsonl.py \
#       --data-dir /path/to/data \
#       --output  aishell_ms_swift.jsonl \
#       --system "You are a helpful ASR system." \
#       --query-prefix "请听下面的语音并给出文字：" \
#       --query-suffix ""                # optional
#
# Each utterance becomes one JSON object.  The audio reference inserted
# into the query is wrapped in <audio> … </audio>.
#
# If a segments file is present, the audio reference is encoded as
#   <wav_path>:<start_sec>:<end_sec>
# so you can decide later how to handle trimming.

import argparse
import json
import os
from pathlib import Path
from typing import Dict, Tuple


def read_wav_scp(path: Path) -> Dict[str, str]:
    """recording-ID  -> wav path / command"""
    mapping = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec_id, wav_path = line.rstrip().split(maxsplit=1)
            mapping[rec_id] = wav_path
    return mapping

def read_text(path: Path) -> Dict[str, str]:
    """utt-ID -> transcript"""
    mapping = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            utt_id, *words = line.rstrip().split()
            mapping[utt_id] = " ".join(words)
    return mapping

def read_segments(path: Path) -> Dict[str, Tuple[str, float, float]]:
    """
    utt-ID -> (recording-ID, start(sec), end(sec))
    segments lines: utt rec start end
    """
    mapping = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            utt_id, rec_id, start, end = line.rstrip().split()
            mapping[utt_id] = (rec_id, float(start), float(end))
    return mapping

def main():
    parser = argparse.ArgumentParser(description="Convert a Kaldi-asr data-dir to ms-swift JSONL")
    parser.add_argument("--data-dir", required=True,
                    help="Path to Kaldi data directory")
    parser.add_argument("--output", required=True,
                    help="Output JSONL file")
    parser.add_argument("--system", required=True,
                    help="Content of the \"system\" field")
    parser.add_argument("--query-prefix", default="",
                    help="Text placed *before* <audio>…</audio>")
    parser.add_argument("--query-suffix", default="",
                    help="Text placed *after* <audio>…</audio>")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    wav_scp = read_wav_scp(data_dir / "wav.scp")
    text_map = read_text(data_dir / "text")

    segments_path = data_dir / "segments"
    use_segments = segments_path.exists()
    seg_map = read_segments(segments_path) if use_segments else {}

    with open(args.output, "w", encoding="utf-8") as fout:
        for utt_id, transcript in text_map.items():
            if use_segments:
                if utt_id not in seg_map:
                    continue  # skip utterances not in segments
                rec_id, start, end = seg_map[utt_id]
                # audio_ref = f"{wav_scp[rec_id]}:{start:.2f}:{end:.2f}"
                audio_ref = wav_scp[rec_id]
            else:
                # wav.scp key must be utt_id when no segments file exists
                if utt_id not in wav_scp:
                    continue  # inconsistent entry, skip
                audio_ref = wav_scp[utt_id]

            query = (
                f"{args.query_prefix}"
                f"<audio>{audio_ref}</audio>"
                f"{args.query_suffix}"
            )
            obj = {
                "system": args.system,
                "query": query,
                "response": transcript
            }
            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")

    print(f"Wrote {args.output}")

if __name__ == "__main__":
    main()