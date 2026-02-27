#!/bin/bash
# =============================================================================
# Qwen3-VL-4B 视觉语言模型监督微调 (SFT) 启动脚本
# 执行流程：本脚本通过 torchrun 启动分布式训练，入口为 train_qwen.py
# =============================================================================

# --------------------------------------------------------------------------
# 【阶段1】分布式训练环境变量配置
# MASTER_ADDR: 主节点地址，默认本机 127.0.0.1（多机训练时需改为主节点 IP）
# MASTER_PORT: 进程通信端口，未指定时在 20001-29999 随机选取以避免冲突
# NNODES:      节点总数，从环境变量 WORLD_SIZE 读取，默认单节点训练
# --------------------------------------------------------------------------
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
MASTER_PORT=${MASTER_PORT:-$(shuf -i 20001-29999 -n 1)}
NNODES=${WORLD_SIZE:-1}

# --------------------------------------------------------------------------
# 【阶段2】DeepSpeed ZeRO 优化配置
# 使用 ZeRO Stage-3 策略：将模型参数、梯度、优化器状态均分片存储到各 GPU
# 大幅降低单卡显存占用，支持在有限显存上训练大模型
# 详细配置见 scripts/zero3.json
# --------------------------------------------------------------------------
deepspeed=./scripts/zero3.json

# --------------------------------------------------------------------------
# 【阶段3】模型配置
# 使用 HuggingFace Hub 上的模型 ID，训练时会自动下载权重
# 也可替换为本地路径，如 /path/to/Qwen3-VL-4B-Instruct
# --------------------------------------------------------------------------
llm=Qwen/Qwen3-VL-4B-Instruct  # Using HuggingFace model ID

# --------------------------------------------------------------------------
# 【阶段4】训练超参数
# lr:              学习率 1e-5，适合微调阶段的较小学习率
# batch_size:      每个 GPU 的 micro batch size = 4
# grad_accum_steps: 梯度累积步数 = 4，等效全局 batch = GPU数 × 4 × 4
# --------------------------------------------------------------------------
lr=1e-5
batch_size=4
grad_accum_steps=4

# --------------------------------------------------------------------------
# 【阶段5】训练入口文件
# torchrun 将启动此 Python 文件，每个 GPU 各启动一个进程
# 详见 qwenvl/train/train_qwen.py
# --------------------------------------------------------------------------
entry_file=qwenvl/train/train_qwen.py

# --------------------------------------------------------------------------
# 【阶段6】数据集配置
# 逗号分隔的数据集名称列表，对应 qwenvl/data/__init__.py 中 data_dict 的键
# 支持采样率语法，如 cambrian_737k%50 表示采样 50%
# --------------------------------------------------------------------------
datasets=public_dataset1,public_dataset2

# --------------------------------------------------------------------------
# 【阶段7】输出配置
# run_name:    实验名称，用于 wandb 日志记录
# output_dir:  模型检查点和最终权重的保存目录
# --------------------------------------------------------------------------
run_name="qwen3vl"
output_dir=./output

# --------------------------------------------------------------------------
# 【阶段8】完整训练参数列表（传递给 train_qwen.py 的命令行参数）
# --deepspeed:                  启用 DeepSpeed，使用指定的 ZeRO 配置文件
# --model_name_or_path:         预训练模型路径或 HF Hub ID
# --dataset_use:                要使用的数据集名称（逗号分隔）
# --data_flatten:               True=使用 packed sequence（变长注意力），减少 padding 浪费
# --tune_mm_vision:             False=冻结 ViT 视觉编码器，不参与训练
# --tune_mm_mlp:                True=训练视觉-语言 MLP Merger 投影层
# --tune_mm_llm:                True=训练 LLM 语言模型主干
# --bf16:                       使用 BFloat16 混合精度训练（需要 Ampere+ GPU）
# --output_dir:                 检查点保存路径
# --num_train_epochs:           训练轮数（0.5 表示只训练半个 epoch，用于快速验证）
# --per_device_train_batch_size: 每卡训练 batch size
# --per_device_eval_batch_size: 每卡评估 batch size（通常是训练的 2 倍）
# --gradient_accumulation_steps: 梯度累积步数，增大等效 batch size
# --max_pixels:                 图像 token 最大像素数（50176 = 224×224）
# --min_pixels:                 图像 token 最小像素数（784 = 28×28）
# --eval_strategy:              评估策略（no=不做中间评估）
# --save_strategy:              保存策略（steps=每隔 N 步保存一次）
# --save_steps:                 每 1000 步保存一次检查点
# --save_total_limit:           最多保留 1 个检查点，节省磁盘空间
# --learning_rate:              AdamW 优化器学习率
# --weight_decay:               L2 正则化权重衰减系数（0=不衰减）
# --warmup_ratio:               学习率 warmup 占总步数比例（3%）
# --max_grad_norm:              梯度裁剪阈值，防止梯度爆炸
# --lr_scheduler_type:          学习率调度器类型（cosine=余弦退火）
# --logging_steps:              每 1 步打印一次日志
# --model_max_length:           最大序列长度（8192 个 token）
# --gradient_checkpointing:     True=梯度检查点，用时间换显存（重计算激活值）
# --dataloader_num_workers:     DataLoader 的并行预取线程数
# --run_name:                   Wandb 实验名称
# --report_to:                  日志上报目标（wandb）
# --------------------------------------------------------------------------
args="
    --deepspeed ${deepspeed} \
    --model_name_or_path "${llm}" \
    --dataset_use ${datasets} \
    --data_flatten True \
    --tune_mm_vision False \
    --tune_mm_mlp True \
    --tune_mm_llm True \
    --bf16 \
    --output_dir ${output_dir} \
    --num_train_epochs 0.5 \
    --per_device_train_batch_size ${batch_size} \
    --per_device_eval_batch_size $((batch_size*2)) \
    --gradient_accumulation_steps ${grad_accum_steps} \
    --max_pixels 50176 \
    --min_pixels 784 \
    --eval_strategy "no" \
    --save_strategy "steps" \
    --save_steps 1000 \
    --save_total_limit 1 \
    --learning_rate ${lr} \
    --weight_decay 0 \
    --warmup_ratio 0.03 \
    --max_grad_norm 1 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --model_max_length 8192 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --run_name ${run_name} \
    --report_to wandb"

# --------------------------------------------------------------------------
# 【阶段9】启动分布式训练
# torchrun 是 PyTorch 官方的分布式启动器（取代旧版 torch.distributed.launch）
# --nproc_per_node: 每个节点启动的进程数（通常等于 GPU 数量）
# --master_addr:    主进程的地址（用于进程间通信 rendezvous）
# --master_port:    主进程的监听端口
# 每个 GPU 进程都会独立执行 train_qwen.py，通过 local_rank 区分身份
# --------------------------------------------------------------------------
torchrun --nproc_per_node=${NPROC_PER_NODE} \
         --master_addr=${MASTER_ADDR} \
         --master_port=${MASTER_PORT} \
         ${entry_file} ${args}