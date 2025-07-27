#!/usr/bin/env bash

# for i in 3 6 9 12 15 18 21 24 27 30; do
#     swift infer \
#         --model Qwen/Qwen2.5-Omni-7B \
#         --ckpt_dir ./output_sia/v0-20250615-110901/checkpoint-${i}/ \
#         --val_dataset ./data_sia/fold1_val.jsonl
# done

for ckpt_dir in  v0-20250705-022121  v1-20250705-022935  v2-20250705-023747 v3-20250705-024634  v4-20250705-025510 ; do
    # Extract fold number from ckpt_dir (e.g., 'v1' -> '1')
    fold_prefix="${ckpt_dir%%-*}"
    fold_num="${fold_prefix#v}"
    val_dataset="./data_sia_batch_1_2_binary_only/fold$((fold_num+1))_val.jsonl"
    for i in 8 16 24 32 40 48 56 64 72 80; do
        swift infer \
            --model Qwen/Qwen2.5-Omni-7B \
            --ckpt_dir "./output_sia_batch_1_2_binary_only/${ckpt_dir}/checkpoint-${i}/" \
            --val_dataset "${val_dataset}"
    done
done

