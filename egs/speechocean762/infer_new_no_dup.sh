#!/usr/bin/env bash

for i in 210; do
    ckpt_dir=./output_new_no_dup/v0-20250619-235922/checkpoint-${i}/

    CUDA_VISIBLE_DEVICES=1 swift infer \
        --model Qwen/Qwen2.5-Omni-7B \
        --ckpt_dir ${ckpt_dir}/ \
        --max_batch_size 4 \
        --val_dataset ./data/test_swift_accuracy.json \
        --result_path ${ckpt_dir}/accuracy.json &

    CUDA_VISIBLE_DEVICES=1 swift infer \
        --model Qwen/Qwen2.5-Omni-7B \
        --ckpt_dir ${ckpt_dir}/ \
        --max_batch_size 4 \
        --val_dataset ./data/test_swift_fluency.json \
        --result_path ${ckpt_dir}/fluency.json 
    wait

    CUDA_VISIBLE_DEVICES=1 swift infer \
        --model Qwen/Qwen2.5-Omni-7B \
        --ckpt_dir ${ckpt_dir}/ \
        --max_batch_size 4 \
        --val_dataset ./data/test_swift_prosodic.json \
        --result_path ${ckpt_dir}/prosodic.json &

    CUDA_VISIBLE_DEVICES=1 swift infer \
        --model Qwen/Qwen2.5-Omni-7B \
        --ckpt_dir ${ckpt_dir}/ \
        --max_batch_size 4 \
        --val_dataset ./data/test_swift_total.json \
        --result_path ${ckpt_dir}/total.json 
    wait
done