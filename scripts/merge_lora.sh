#!/bin/bash

MODEL_NAME="scb10x/typhoon2-qwen2vl-7b-vision-instruct"
# MODEL_NAME="Qwen/Qwen2-VL-2B-Instruct"

export PYTHONPATH=src:$PYTHONPATH

python src/merge_lora_weights.py \
    --model-path /home/workspace/Qwen2-VL-Finetune/output/testing_lora \
    --model-base $MODEL_NAME  \
    --save-model-path /home/workspace/Qwen2-VL-Finetune/output/merge_test \
    --safe-serialization