#!/usr/bin/env bash

swift sft \
    --model Qwen/Qwen2.5-Omni-7B \
    --dataset ./data/train_swift.json \
    --val_dataset ./data/dev_swift.json \
    --train_type lora \
    --torch_dtype bfloat16 \
    --num_train_epochs 5 \
    --per_device_train_batch_size 128 \
    --per_device_eval_batch_size 128 \
    --learning_rate 1e-4 \
    --lora_rank 8 \
    --lora_alpha 32 \
    --target_modules all-linear \
    --freeze_vit false \
    --gradient_accumulation_steps 1 \
    --eval_strategy epoch \
    --save_strategy epoch \
    --save_total_limit -1 \
    --logging_steps 5 \
    --max_length 2048 \
    --output_dir output_new_unfreeze_vit \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 16