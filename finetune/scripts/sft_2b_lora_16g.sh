#!/bin/bash
# ============================================================================
# Qwen3-VL-2B LoRA Fine-tuning for Visual Grounding
# Configuration: Single GPU with 16GB VRAM (e.g., RTX 4080/4090 16GB)
#
# Key optimizations for 16GB:
#   - batch_size=1 with grad_accum=16 (effective batch=16)
#   - gradient_checkpointing=True (saves ~40% VRAM)
#   - smaller max_pixels (reduces image token count)
#   - DeepSpeed ZeRO-2 (offloads optimizer states)
# ============================================================================

# Distributed training configuration (single GPU)
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
MASTER_PORT=${MASTER_PORT:-$(shuf -i 20001-29999 -n 1)}
NPROC_PER_NODE=${NPROC_PER_NODE:-1}

# DeepSpeed configuration
deepspeed=./scripts/zero2.json

# Model configuration — change this to your local checkpoint path if needed
llm=../checkpoint/Qwen3-VL-2B-Instruct

# Training hyperparameters — conservative for 16GB VRAM
lr=2e-5
batch_size=16
grad_accum_steps=1

# Training entry point
entry_file=qwenvl/train/train_qwen.py

# Dataset configuration
datasets=refcoco_grounding
eval_datasets=refcoco_grounding_val

# Output configuration
run_name="qwen3vl_2b_lora_grounding_16g"
output_dir=./output/grounding_lora_16g

# Training arguments
args="
    --deepspeed ${deepspeed} \
    --model_name_or_path ${llm} \
    --dataset_use ${datasets} \
    --eval_dataset_use ${eval_datasets} \
    --data_flatten True \
    --tune_mm_vision False \
    --tune_mm_mlp False \
    --tune_mm_llm True \
    --bf16 \
    --output_dir ${output_dir} \
    --num_train_epochs 50 \
    --per_device_train_batch_size ${batch_size} \
    --per_device_eval_batch_size ${batch_size} \
    --gradient_accumulation_steps ${grad_accum_steps} \
    --max_pixels 50176 \
    --min_pixels 784 \
    --eval_strategy epoch \
    --save_strategy steps \
    --save_steps 200 \
    --save_total_limit 2 \
    --learning_rate ${lr} \
    --weight_decay 0.01 \
    --warmup_ratio 0.05 \
    --max_grad_norm 1 \
    --lr_scheduler_type cosine \
    --logging_steps 5 \
    --model_max_length 8192 \
    --gradient_checkpointing False \
    --dataloader_num_workers 4 \
    --run_name ${run_name} \
    --report_to wandb \
    --lora_enable True \
    --lora_r 64 \
    --lora_alpha 128 \
    --lora_dropout 0.05"

# Launch training
torchrun --nproc_per_node=${NPROC_PER_NODE} \
         --master_addr=${MASTER_ADDR} \
         --master_port=${MASTER_PORT} \
         ${entry_file} ${args}
