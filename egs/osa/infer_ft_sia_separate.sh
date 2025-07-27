#!/usr/bin/env bash

# for i in 3 6 9 12 15 18 21 24 27 30; do
#     swift infer \
#         --model Qwen/Qwen2.5-Omni-7B \
#         --ckpt_dir ./output_sia/v0-20250615-110901/checkpoint-${i}/ \
#         --val_dataset ./data_sia/fold1_val.jsonl
# done

i=3
for unit in pinyin ; do
    val_dataset="./data_sia_${unit}/test.jsonl"
    swift infer \
        --model Qwen/Qwen2.5-Omni-7B \
        --ckpt_dir "./output_sia_${unit}/v0-20250617-191711/checkpoint-${i}/" \
        --val_dataset "${val_dataset}"
done

for unit in char ; do
    val_dataset="./data_sia_${unit}/test.jsonl"
    swift infer \
        --model Qwen/Qwen2.5-Omni-7B \
        --ckpt_dir "./output_sia_${unit}/v0-20250617-191341/checkpoint-${i}/" \
        --val_dataset "${val_dataset}"
done

for unit in sentence ; do
    val_dataset="./data_sia_${unit}/test.jsonl"
    swift infer \
        --model Qwen/Qwen2.5-Omni-7B \
        --ckpt_dir "./output_sia_${unit}/v0-20250617-192022/checkpoint-${i}/" \
        --val_dataset "${val_dataset}"
done