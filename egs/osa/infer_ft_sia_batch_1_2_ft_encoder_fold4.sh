#!/usr/bin/env bash

# for i in 3 6 9 12 15 18 21 24 27 30; do
#     swift infer \
#         --model Qwen/Qwen2.5-Omni-7B \
#         --ckpt_dir ./output_sia/v0-20250615-110901/checkpoint-${i}/ \
#         --val_dataset ./data_sia/fold1_val.jsonl
# done

for ckpt_dir in v0-20250706-004649 ; do
    # Extract fold number from ckpt_dir (e.g., 'v1' -> '1')
    # fold_prefix="${ckpt_dir%%-*}"
    # fold_num="${fold_prefix#v}"
    val_dataset="./data_sia_batch_1_2/fold4_val.jsonl"
    for i in 8 16 24 32 40 48 56 64 72 80 88 96 104 112 120; do
        swift infer \
            --model Qwen/Qwen2.5-Omni-7B \
            --ckpt_dir "./output_sia_batch_1_2_round_number_ft_encoder/ft_fold4/${ckpt_dir}/checkpoint-${i}/" \
            --val_dataset "${val_dataset}"
    done
done

