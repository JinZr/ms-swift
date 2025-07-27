#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=1 swift infer \
    --model Qwen/Qwen2.5-Omni-7B \
    --ckpt_dir ./output_new/v0-20250619-142550/checkpoint-252/ \
    --val_dataset ./data/test_swift_accuracy.json \
    --result_path ./output_new/v0-20250619-142550/checkpoint-252/accuracy.json &

CUDA_VISIBLE_DEVICES=1 swift infer \
    --model Qwen/Qwen2.5-Omni-7B \
    --ckpt_dir ./output_new/v0-20250619-142550/checkpoint-252/ \
    --val_dataset ./data/test_swift_fluency.json \
    --result_path ./output_new/v0-20250619-142550/checkpoint-252/fluency.json &
wait

CUDA_VISIBLE_DEVICES=1 swift infer \
    --model Qwen/Qwen2.5-Omni-7B \
    --ckpt_dir ./output_new/v0-20250619-142550/checkpoint-252/ \
    --val_dataset ./data/test_swift_prosodic.json \
    --result_path ./output_new/v0-20250619-142550/checkpoint-252/prosodic.json &

CUDA_VISIBLE_DEVICES=1 swift infer \
    --model Qwen/Qwen2.5-Omni-7B \
    --ckpt_dir ./output_new/v0-20250619-142550/checkpoint-252/ \
    --val_dataset ./data/test_swift_total.json \
    --result_path ./output_new/v0-20250619-142550/checkpoint-252/total.json &
wait