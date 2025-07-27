#!/usr/bin/env python3
"""
Convert SHI-style annotation JSON to conversational SFT format.

Usage:
    python convert_for_sft.py infile.json outfile.json

This version additionally maps fine‑grained error labels in `csv_mistake` to their general categories.
"""
import argparse
import json
import re
from pathlib import Path

# === 可自行修改的模板 ===
SYSTEM_PROMPT = (
    "你是一名语言病理学专家，专长于识别儿童及成人的构音错误。"
    "请结合患者朗读文本与实际发音，对发音问题做出专业判断。"
)

QUERY_TEMPLATE = (
    "以下是一段朗读录音，请根据音频内容列出出现了的构音错误，"
    "朗读文本：{transcript}\n"
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

def map_to_error_category(fine_error: str) -> str:
    """
    将细粒度构音错误标签映射到两大类别，并支持模糊匹配。

    参数
    ----
    fine_error : str
        输入的细粒度构音错误（允许简写，如“侧化”“低压力”等）。

    返回
    ----
    str
        “发育型构音错误” / “腭裂特征型构音错误” / “未知类别”
    """
    if not fine_error:  # 空字符串处理
        raise KeyError(fine_error)

    fine_error = fine_error.strip()          # 去除首尾空格
    # -------- 1. 维护「标签 → 大类」的映射 --------
    label2category = {
        # 发育型
        "前置构音": "发育型构音错误",
        "双唇音": "发育型构音错误", # not sure
        "双唇音化": "发育型构音错误", # not sure
        "舌面音化": "发育型构音错误",
        "舌尖前音化": "发育型构音错误",
        "舌尖后音化": "发育型构音错误",
        "后置构音": "发育型构音错误",        # 注意：也出现在腭裂特征型，稍后冲突解决
        "侧化构音": "发育型构音错误",
        "塞音化": "发育型构音错误",
        "擦音化": "发育型构音错误",
        "塞擦音化": "发育型构音错误",
        "不送气化": "发育型构音错误",
        "声随韵母省略": "发育型构音错误",
        "复韵母简化": "发育型构音错误",
        "介音省略": "发育型构音错误",

        # 腭裂特征型
        "喉塞音": "腭裂特征型构音错误",
        "咽擦音": "腭裂特征型构音错误",
        "咽塞擦音": "腭裂特征型构音错误",
        "咽塞音": "腭裂特征型构音错误",
        "鼻腔擦音": "腭裂特征型构音错误",
        "代偿性构音": "腭裂特征型构音错误",
        "咽部代偿": "腭裂特征型构音错误", # not sure
        "鼻音化": "腭裂特征型构音错误",
        "低压力辅音": "腭裂特征型构音错误",
        "低压力构音": "腭裂特征型构音错误", # not sure
        "声母省略": "腭裂特征型构音错误",
        "软腭擦音": "腭裂特征型构音错误",
        "腭化构音": "腭裂特征型构音错误",
        "口腔后置构音": "腭裂特征型构音错误",
        # 冲突标签：在腭裂分类体系中“后置构音”常指代偿性 / 生理限制导致的后置，
        # 若希望优先归到腭裂特征型，可在此覆盖：
        "后置构音(腭裂特征)": "腭裂特征型构音错误",
    }

    # -------- 2. 先尝试精准匹配 --------
    if fine_error in label2category:
        return label2category[fine_error]

    # -------- 3. 模糊匹配（包含关系） --------
    for label, category in label2category.items():
        if fine_error in label or label in fine_error:
            return category

    raise KeyError(fine_error)


# === 新增辅助函数: 细粒度错误标签映射 ===
def map_fine_errors_in_str(mistake_str: str) -> str:
    """
    将 csv_mistake 中的细粒度错误标签映射为通用类别。
    支持处理形如 "(谢,咽擦音) (弟,后置+鼻音化)" 的复合格式，
    并对单独字符串（如 "低压力构音"）进行同样映射。

    规则：
    1. 对于括号表达式，保留原先的第一个字段（字或序号），
       仅将错误标签部分替换为对应的大类。
    2. 若错误标签中包含 “+”/“＋”，先拆分再逐个映射，
       最终用 "+" 连接去重后的大类列表，保持稳定排序。
    """
    if not mistake_str or mistake_str.strip() == "":
        return "没有构音错误"

    # 若字符串中没有括号，视为单一/复合标签
    if "(" not in mistake_str:
        parts = re.split(r"[+＋]", mistake_str)
        cats = []
        for p in parts:
            p = p.strip()
            try:
                cat = map_to_error_category(p)
                if cat not in cats:
                    cats.append(cat)
            except KeyError:
                print(p)
        # return "+".join(cats)
        return " ".join(cats)

    # 处理括号模式
    pattern = re.compile(r"\(([^,]+),\s*([^)]+)\)")
    converted_chunks = []
    for m in pattern.finditer(mistake_str):
        char_or_idx = m.group(1).strip()
        fine_label_raw = m.group(2).strip()

        sub_labels = re.split(r"[+＋]", fine_label_raw)
        mapped = []
        for sub in sub_labels:
            sub = sub.strip()
            try:
                cat = map_to_error_category(sub)
                if cat not in mapped:
                    mapped.append(cat)
            except KeyError:
                print(sub_labels)

        converted_chunks.append(f"{' '.join(mapped)}")

    return " ".join(converted_chunks)

def main(in_file: Path, out_file: Path):
    with in_file.open("r", encoding="utf-8") as f:
        data = json.load(f)

    converted = []
    for it in data:
        # 原始 csv_mistake 字段标准化
        mistake_field = it.get("csv_mistake", "")
        if mistake_field in ("", "正常"):
            it["csv_mistake"] = "没有构音错误"
        else:
            it["csv_mistake"] = map_fine_errors_in_str(mistake_field)

        converted_item = convert_item(it)
        if converted_item:
            converted.append(converted_item)

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