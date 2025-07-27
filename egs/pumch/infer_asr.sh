#!/usr/bin/env bash

# swift infer \
#     --model_type qwen2-audio-7b-instruct \
#     --model_id_or_path /home/jinzr/.cache/modelscope/hub/qwen/Qwen2-Audio-7B-Instruct/ \
#     --ckpt_dir ./output/qwen2-audio-7b-instruct/v0-20250613-023911/checkpoint-720/ \
#     --merge_lora true \
#     --val_dataset ./data/fold1_val.jsonl 

swift infer \
    --model Qwen/Qwen2.5-Omni-7B \
    --ckpt_dir ./output_asr/v0-20250713-202652/checkpoint-78 \
    --val_dataset ./data/SHI_batch_2_swift.json
