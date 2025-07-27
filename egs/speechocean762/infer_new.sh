#!/usr/bin/env bash

for i in 50 100 150 200; do
    ckpt_dir=./output_new_new/v2-20250622-183032/checkpoint-${i}/

    CUDA_VISIBLE_DEVICES=5 swift infer \
        --model Qwen/Qwen2.5-Omni-7B \
        --ckpt_dir ${ckpt_dir}/ \
        --val_dataset ./new_data/test_swift_accuracy.json \
        --result_path ${ckpt_dir}/accuracy.json &

    CUDA_VISIBLE_DEVICES=5 swift infer \
        --model Qwen/Qwen2.5-Omni-7B \
        --ckpt_dir ${ckpt_dir}/ \
        --val_dataset ./new_data/test_swift_fluency.json \
        --result_path ${ckpt_dir}/fluency.json 
    wait

    CUDA_VISIBLE_DEVICES=5 swift infer \
        --model Qwen/Qwen2.5-Omni-7B \
        --ckpt_dir ${ckpt_dir}/ \
        --val_dataset ./new_data/test_swift_prosodic.json \
        --result_path ${ckpt_dir}/prosodic.json &

    CUDA_VISIBLE_DEVICES=5 swift infer \
        --model Qwen/Qwen2.5-Omni-7B \
        --ckpt_dir ${ckpt_dir}/ \
        --val_dataset ./new_data/test_swift_total.json \
        --result_path ${ckpt_dir}/total.json 
    wait
done