import json
import pandas as pd
import re
from collections import defaultdict
from scipy.stats import pearsonr

# --------------------------------------------------
# 1. CONFIG
# --------------------------------------------------
FILE_PATH = "/home/jinzr/nfs/projects/ms-swift/egs/osa/output/qwen2-audio-7b-instruct/v0-20250613-023911/checkpoint-720-merged/infer_result/20250613-124129.jsonl"

# --------------------------------------------------
# 2. 工具函数
# --------------------------------------------------
category_re = re.compile(r"估算说话人的(.*?):")


def extract_category(query: str) -> str | None:
    """从 query 中提取预测任务类别 如 AHI、BMI"""
    m = category_re.search(query)
    return m.group(1).strip() if m else None


def extract_value(text: str) -> str:
    """提取响应或标签中的数值/等级部分"""
    m = re.search(r"[:：]\s*(.*)$", text)
    return m.group(1).strip() if m else text.strip()


def to_float(val: str) -> float | None:
    """尝试将文本转换为 float；无法转换时返回 None"""
    try:
        return float(val.replace("%", ""))
    except ValueError:
        print(val)
        return None


# 罗马数字映射→阿拉伯数字，便于比较
roman_map = {"I": "1", "II": "2", "III": "3", "IV": "4", "V": "5"}

# 音频索引 → 说话人位置/读法
index_to_pos = {
    "1": "坐位拼音",
    "2": "坐位汉字",
    "3": "坐位句子",
    "4": "仰卧位拼音",
    "5": "仰卧位汉字",
    "6": "仰卧位句子",
}

# --------------------------------------------------
# 3. 读取 JSONL 并按类别分组
# --------------------------------------------------
pairs = defaultdict(
    lambda: defaultdict(list)
)  # {category: {position: [(pred,true),...]}}

with open(FILE_PATH, "r", encoding="utf-8") as f:
    for line in f:
        if not line.strip():
            continue
        rec = json.loads(line)
        cat = extract_category(rec["query"])
        if not cat:
            continue
        # --- 提取说话人位置 ---
        pos = None
        # 1) 先尝试根据音频文件名索引判断
        wav_idx = re.search(r"_([1-6])\.wav", rec["query"])
        if wav_idx:
            pos = index_to_pos.get(wav_idx.group(1))
        # 2) 若未命中，再从文字描述中粗略提取
        if not pos:
            m_pos = re.search(r"(坐位|仰卧位|卧位).*?(拼音|汉字|句子)", rec["query"])
            if m_pos:
                first = "坐位" if m_pos.group(1).startswith("坐") else "仰卧位"
                pos = f"{first}{m_pos.group(2)}"
        if not pos:
            pos = "未知位置"
        pred = extract_value(rec["response"])
        label = extract_value(rec["label"])
        pairs[cat][pos].append((pred, label))

# --------------------------------------------------
# 4. 计算 Pearson r 或准确率
# --------------------------------------------------
results = []

for cat, pos_dict in pairs.items():
    for pos, pair_list in pos_dict.items():
        if "分级" in cat:  # 分类任务
            correct = sum(
                1
                for pred, true in pair_list
                if roman_map.get(pred.strip().upper(), pred.strip())
                == roman_map.get(true.strip().upper(), true.strip())
            )
            total = len(pair_list)
            acc = correct / total if total else float("nan")
            results.append(
                {
                    "category": cat,
                    "position": pos,
                    "metric": "accuracy",
                    "value": acc,
                    "n": total,
                }
            )
        else:  # 回归任务
            preds, trues = (
                zip(
                    *[
                        (to_float(pred), to_float(true))
                        for pred, true in pair_list
                        if to_float(pred) is not None and to_float(true) is not None
                    ]
                )
                if pair_list
                else ([], [])
            )
            if len(set(preds)) > 1 and len(set(trues)) > 1 and len(preds) >= 2:
                r, _ = pearsonr(preds, trues)
            else:
                print(preds, trues, cat)
                r = float("nan")
            results.append(
                {
                    "category": cat,
                    "position": pos,
                    "metric": "pearson_r",
                    "value": r,
                    "n": len(preds),
                }
            )

# --------------------------------------------------
# 5. 输出 DataFrame（可选）
# --------------------------------------------------
df = pd.DataFrame(results).sort_values(["category", "position"]).reset_index(drop=True)
# 打印完整 DataFrame，不折叠显示
with pd.option_context(
    "display.max_rows",
    None,
    "display.max_columns",
    None,
    "display.width",
    None,
    "display.max_colwidth",
    None,
):
    print(df)
