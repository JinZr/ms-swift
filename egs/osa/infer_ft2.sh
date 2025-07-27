#!/usr/bin/env bash

swift infer \
    --model_type qwen2-audio-7b-instruct \
    --model_id_or_path /home/jinzr/.cache/modelscope/hub/qwen/Qwen2-Audio-7B-Instruct/ \
    --ckpt_dir ./output/qwen2-audio-7b-instruct/v1-20250613-165913/checkpoint-270/ \
    --merge_lora true \
    --val_dataset ./data2/fold1_val.jsonl 
