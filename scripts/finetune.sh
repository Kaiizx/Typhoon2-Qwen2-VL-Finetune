#!/bin/bash

# You can use 2B instead of 7B
# MODEL_NAME="Qwen/Qwen2-VL-7B-Instruct"
# MODEL_NAME="scb10x/typhoon2-qwen2vl-7b-vision-instruct"
MODEL_NAME="Qwen/Qwen2-VL-2B-Instruct"

GLOBAL_BATCH_SIZE=1024
BATCH_PER_DEVICE=128
NUM_DEVICES=4
GRAD_ACCUM_STEPS=$((GLOBAL_BATCH_SIZE / (BATCH_PER_DEVICE * NUM_DEVICES)))

export PYTHONPATH=src:$PYTHONPATH

accelerate launch \
    --num_processes $NUM_DEVICES \
    --num_machines 1 \
    --mixed_precision bf16 \
    --multi_gpu \
    --dynamo_backend inductor \
    src/training/train.py \
        --deepspeed scripts/zero3.json \
        --model_id $MODEL_NAME \
        --data_path /lustre/scratch/public/AI-cooking-hack/captioning/capgen_llava_train.json \
        --image_folder /lustre/scratch/public/AI-cooking-hack/captioning \
        --freeze_vision_tower True \
        --freeze_llm True \
        --tune_merger True \
        --bf16 True \
        --fp16 False \
        --disable_flash_attn2 False \
        --output_dir output/qwen2-2b-merger-ft \
        --num_train_epochs 3 \
        --per_device_train_batch_size $BATCH_PER_DEVICE \
        --gradient_accumulation_steps $GRAD_ACCUM_STEPS \
        --min_pixels $((256 * 28 * 28)) \
        --max_pixels $((1280 * 28 * 28)) \
        --learning_rate 1e-5 \
        --merger_lr 1e-5 \
        --vision_lr 2e-6 \
        --weight_decay 0.1 \
        --warmup_ratio 0.03 \
        --adam_beta2 0.95 \
        --lr_scheduler_type "cosine" \
        --logging_steps 100 \
        --tf32 True \
        --gradient_checkpointing True \
        --report_to 'none' \
        --lazy_preprocess True \
        --save_strategy "epoch" \
        --dataloader_num_workers 16