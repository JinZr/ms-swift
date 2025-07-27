#!/usr/bin/env python3
"""
Convert SHI-style annotation JSON to conversational SFT format.

Usage:
    python convert_for_sft.py infile.json outfile.json
"""
import argparse
import json
from pathlib import Path

# === 可自行修改的模板 ===
SYSTEM_PROMPT = (
    "你是一名语言病理学专家，专长于识别儿童及成人的构音错误。"
    "请结合患者朗读文本与实际发音，对发音问题做出专业判断。"
)

QUERY_TEMPLATE = (
    "以下是一段朗读录音，请根据音频内容列出每个字的构音错误，"
    "朗读文本：{transcript}"
    "音频文件：<audio>{wav_path}</audio>"
)

def convert_item(item: dict) -> dict | None:
    """Return converted dict if wav_path exists, else None."""
    wav = item.get("wav_path")
    if not wav:
        return None  # 无音频时跳过
    query = QUERY_TEMPLATE.format(
        transcript=item.get("kaldi_transcript", ""),
        wav_path=wav
    )
    return {
        "system": SYSTEM_PROMPT,
        "query": query,
        "response": item.get("csv_mistake", "")
    }

def main(in_file: Path, out_file: Path):
    with in_file.open("r", encoding="utf-8") as f:
        data = json.load(f)

    converted = []
    for it in data:
        if convert_item(it):
            if it["csv_mistake"] != "":
                it["csv_mistake"] = "没有构音错误" if it["csv_mistake"] == "正常" else it["csv_mistake"]
                converted.append(convert_item(it))
            else:
                it["csv_mistake"] = "没有构音错误"
                converted.append(convert_item(it))


    # 保持可读性，使用缩进 2；如要生成 jsonl，更换写入逻辑即可
    with out_file.open("w", encoding="utf-8") as f:
        json.dump(converted, f, ensure_ascii=False, indent=2)

    print(f"Converted {len(converted)} items ➜ {out_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert SHI JSON to SFT format")
    parser.add_argument("infile", type=Path, help="原始 JSON 文件")
    parser.add_argument("outfile", type=Path, help="输出 JSON 文件")
    args = parser.parse_args()
    main(args.infile, args.outfile)