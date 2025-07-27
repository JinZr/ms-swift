#!/usr/bin/env bash
# lora
swift sft \
    --model Qwen/Qwen2.5-Omni-7B \
    --dataset ./new_data_cot_interleave/train_swift.json \
    --val_dataset ./new_data_cot_interleave/dev_swift.json \
    --train_type lora \
    --torch_dtype float32 \
    --num_train_epochs 5 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 16 \
    --learning_rate 1e-6 \
    --lora_rank 8 \
    --lora_alpha 32 \
    --lora_dropout  0.05  \
    --target_modules all-linear \
    --freeze_vit true \
    --gradient_accumulation_steps 1 \
    --eval_strategy epoch \
    --save_strategy epoch \
    --save_total_limit -1 \
    --logging_steps 5 \
    --max_length 2048 \
    --output_dir output_new_cot_interleave \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 16