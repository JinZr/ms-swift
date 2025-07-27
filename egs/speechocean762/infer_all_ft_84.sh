#!/usr/bin/env bash

for i in 84 168 252 336 420; do
    ckpt_dir=./output_new_unfreeze_vit/v0-20250619-144030/checkpoint-${i}

    swift infer \
        --model Qwen/Qwen2.5-Omni-7B \
        --ckpt_dir ${ckpt_dir}/ \
        --val_dataset ./data/test_swift_accuracy.json \
        --result_path ${ckpt_dir}/accuracy.json &

    swift infer \
        --model Qwen/Qwen2.5-Omni-7B \
        --ckpt_dir ${ckpt_dir}/ \
        --val_dataset ./data/test_swift_fluency.json \
        --result_path ${ckpt_dir}/fluency.json 
    wait

    swift infer \
        --model Qwen/Qwen2.5-Omni-7B \
        --ckpt_dir ${ckpt_dir}/ \
        --val_dataset ./data/test_swift_prosodic.json \
        --result_path ${ckpt_dir}/prosodic.json &

    swift infer \
        --model Qwen/Qwen2.5-Omni-7B \
        --ckpt_dir ${ckpt_dir}/ \
        --val_dataset ./data/test_swift_total.json \
        --result_path ${ckpt_dir}/total.json 
    wait
done