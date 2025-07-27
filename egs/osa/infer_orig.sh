#!/usr/bin/env bash

swift infer \
    --model_type qwen2-audio-7b-instruct \
    --val_dataset ./data/fold1_val.jsonl \
    --result_dir ./output/qwen2-audio-7b-instruct/origin/ 