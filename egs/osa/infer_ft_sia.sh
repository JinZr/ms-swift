#!/usr/bin/env bash

# for i in 3 6 9 12 15 18 21 24 27 30; do
#     swift infer \
#         --model Qwen/Qwen2.5-Omni-7B \
#         --ckpt_dir ./output_sia/v0-20250615-110901/checkpoint-${i}/ \
#         --val_dataset ./data_sia/fold1_val.jsonl
# done

for ckpt_dir in v1-20250615-111339 v2-20250615-111806 v3-20250615-112241 v4-20250615-112714; do
    # Extract fold number from ckpt_dir (e.g., 'v1' -> '1')
    fold_prefix="${ckpt_dir%%-*}"
    fold_num="${fold_prefix#v}"
    val_dataset="./data_sia/fold${fold_num}_val.jsonl"
    for i in 3 6 9 12 15 18 21 24 27 30; do
        swift infer \
            --model Qwen/Qwen2.5-Omni-7B \
            --ckpt_dir "./output_sia/${ckpt_dir}/checkpoint-${i}/" \
            --val_dataset "${val_dataset}"
    done
done