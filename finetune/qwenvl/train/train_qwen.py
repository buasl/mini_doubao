# Adopted from https://github.com/lm-sys/FastChat. Below is the original copyright:
# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import os
import logging
import pathlib
import torch
import transformers
import sys
from pathlib import Path

# 将项目根目录加入 Python 路径，以便 import qwenvl.* 和 trainer 模块
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

# 从 trainer.py 导入注意力替换函数（用于支持 packed sequence 的变长 FlashAttention）
from trainer import replace_qwen2_vl_attention_class

# 导入四种 Qwen VL 模型类，根据模型名称自动选择合适的类
from transformers import (
    Qwen2VLForConditionalGeneration,        # Qwen2-VL 系列
    Qwen2_5_VLForConditionalGeneration,     # Qwen2.5-VL 系列
    Qwen3VLForConditionalGeneration,        # Qwen3-VL 密集模型
    Qwen3VLMoeForConditionalGeneration      # Qwen3-VL MoE 混合专家模型
)
# 数据模块构建函数：返回 train_dataset、eval_dataset、data_collator
from qwenvl.data.data_processor import make_supervised_data_module
# 三组训练参数 dataclass（见 argument.py）
from qwenvl.train.argument import (
    ModelArguments,
    DataArguments,
    TrainingArguments,
)
# AutoProcessor: 统一的处理器，包含 tokenizer + image_processor + video_processor
# Trainer: HuggingFace 标准训练器（已在 trainer.py 中被 monkey patch）
from transformers import AutoProcessor, Trainer

# 全局变量：保存当前进程的 local rank（0 = 主进程，用于控制日志输出）
local_rank = None


def rank0_print(*args):
    """仅在主进程（rank 0）上打印日志，避免分布式训练时重复输出"""
    if local_rank == 0:
        print(*args)


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """安全地将模型权重收集并保存到磁盘。
    
    DeepSpeed ZeRO-3 模式下参数被分片到各 GPU，需要先聚合再保存。
    非 DeepSpeed 模式下，将 state_dict 搬到 CPU 后保存，避免显存占用。
    """
    if trainer.deepspeed:
        # ZeRO-3 模式：先同步 CUDA，再通过 DeepSpeed 的 save_model 聚合分片参数
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        return

    # 非 DeepSpeed 模式：手动搬运 state_dict 到 CPU 后保存
    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:  # 只在需要保存的进程（通常是 rank 0）执行
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict  # 释放 GPU 显存
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def set_model(model_args, model):
    """根据 ModelArguments 中的 tune_* 标志，精细控制各模块参数是否参与梯度更新。
    
    模型结构划分：
      model.visual            → ViT 视觉编码器（包含 blocks + merger）
      model.visual.merger     → MLP Merger 投影层（单独控制）
      model.language_model    → LLM 语言模型主干（Transformer decoder）
      model.lm_head           → 语言模型输出头（词表投影）
    """
    # --- 控制 ViT 视觉编码器 blocks 的梯度 ---
    if model_args.tune_mm_vision:
        for n, p in model.visual.named_parameters():
            p.requires_grad = True   # 解冻：ViT 全部参数可训练
    else:
        for n, p in model.visual.named_parameters():
            p.requires_grad = False  # 冻结：ViT 全部参数不参与训练

    # --- 单独控制 MLP Merger 投影层的梯度（覆盖上面的 visual 设置）---
    # 注意：merger 是 visual 的子模块，这里单独覆写允许「冻结 ViT 但训练 Merger」
    if model_args.tune_mm_mlp:
        for n, p in model.visual.merger.named_parameters():
            p.requires_grad = True   # 解冻 Merger
    else:
        for n, p in model.visual.merger.named_parameters():
            p.requires_grad = False  # 冻结 Merger

    # --- 控制 LLM 语言模型主干的梯度 ---
    if model_args.tune_mm_llm:
        for n, p in model.language_model.named_parameters():
            p.requires_grad = True
        model.lm_head.requires_grad = True   # 同步解冻输出头
    else:
        for n, p in model.language_model.named_parameters():
            p.requires_grad = False
        model.lm_head.requires_grad = False  # 同步冻结输出头


def train(attn_implementation="flash_attention_2"):
    """训练主函数：完整的 SFT 训练流程。
    
    attn_implementation: 注意力实现方式，默认使用 FlashAttention-2（需要 CUDA 支持）。
    
    执行阶段：
      1. 参数解析
      2. 模型加载与类型检测
      3. Processor / Tokenizer 加载
      4. 注意力机制替换（可选，用于 packed sequence）
      5. 梯度检查点配置
      6. 参数冻结/解冻设置
      7. 数据模块构建
      8. Trainer 初始化与训练
      9. 模型保存
    """
    global local_rank

    # -----------------------------------------------------------------------
    # 【阶段1】解析命令行参数
    # HfArgumentParser 将 sys.argv 中的 --xxx 参数解析映射到三个 dataclass 实例
    # 对应 sft_qwen3_4b.sh 中 args 变量传入的所有参数
    # -----------------------------------------------------------------------
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # 从 TrainingArguments 获取当前进程的 local_rank（每个 GPU 一个进程）
    local_rank = training_args.local_rank
    # 确保输出目录存在
    os.makedirs(training_args.output_dir, exist_ok=True)

    # -----------------------------------------------------------------------
    # 【阶段2】根据模型名称检测类型并加载预训练权重
    # 支持四种模型族：
    #   - Qwen3-VL MoE（模型名含 "qwen3" 且含 "a"，如 Qwen3-VL-30B-A3B）
    #   - Qwen3-VL 密集模型（如 Qwen3-VL-4B-Instruct）
    #   - Qwen2.5-VL（如 Qwen2.5-VL-7B-Instruct）
    #   - Qwen2-VL（旧版基线）
    # dtype=bfloat16 与 --bf16 参数对应，减少显存占用并加速训练
    # -----------------------------------------------------------------------
    if "qwen3" in model_args.model_name_or_path.lower() and "a" in Path(model_args.model_name_or_path.rstrip("/")).name.lower():
        # Qwen3-VL MoE 模型（混合专家架构，模型名中含 "a" 表示 active 参数量）
        model = Qwen3VLMoeForConditionalGeneration.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            attn_implementation=attn_implementation,
            dtype=(torch.bfloat16 if training_args.bf16 else None),
        )
        data_args.model_type = "qwen3vl"  # 动态设置模型类型，供数据处理选择 RoPE 函数
    elif "qwen3" in model_args.model_name_or_path.lower():
        # Qwen3-VL 密集模型（本脚本 sft_qwen3_4b.sh 使用的路径）
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            attn_implementation=attn_implementation,
            dtype=(torch.bfloat16 if training_args.bf16 else None),
        )
        data_args.model_type = "qwen3vl"
    elif "qwen2.5" in model_args.model_name_or_path.lower():
        # Qwen2.5-VL 系列
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            attn_implementation=attn_implementation,
            dtype=(torch.bfloat16 if training_args.bf16 else None),
        )
        data_args.model_type = "qwen2.5vl"
    else:
        # 默认回退到 Qwen2-VL
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            attn_implementation=attn_implementation,
            dtype=(torch.bfloat16 if training_args.bf16 else None),
        )
        data_args.model_type = "qwen2vl"

    print(f'the initlized model is {model_args.model_name_or_path} the class is {model.__class__.__name__}')

    # -----------------------------------------------------------------------
    # 【阶段3】加载 AutoProcessor（统一封装 tokenizer + image_processor + video_processor）
    # processor 负责将原始图像/视频/文本转换为模型输入张量
    # -----------------------------------------------------------------------
    processor = AutoProcessor.from_pretrained(
        model_args.model_name_or_path,
    )

    # -----------------------------------------------------------------------
    # 【阶段4】注意力机制替换（Monkey Patch）
    # 当启用 data_flatten 或 data_packing 时，多个样本被拼接成一条长序列
    # 此时需要变长注意力（varlen Flash Attention）以正确区分样本边界
    # replace_qwen2_vl_attention_class() 将官方 attention 的 forward 替换为
    # 支持 cu_seqlens 累积长度掩码的自定义实现（见 trainer.py）
    # -----------------------------------------------------------------------
    if data_args.data_flatten or data_args.data_packing:
        replace_qwen2_vl_attention_class()
    # 训练阶段禁用 KV Cache（KV Cache 用于推理加速，训练时无需）
    model.config.use_cache = False

    # -----------------------------------------------------------------------
    # 【阶段5】梯度检查点（Gradient Checkpointing）配置
    # 原理：正向传播时不保存中间激活值，反向传播时重新计算
    # 代价：训练速度约降低 30%；收益：显著减少激活值显存占用
    # -----------------------------------------------------------------------
    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            # HuggingFace 模型内置接口：启用输入嵌入层的梯度计算
            model.enable_input_require_grads()
        else:
            # 回退方案：通过 forward hook 强制输出 requires_grad=True
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)

            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    # -----------------------------------------------------------------------
    # 【阶段6a】加载 Tokenizer
    # padding_side="right": 右填充，与 causal LM 训练对齐
    # use_fast=False: 使用 Python 实现的 tokenizer（慢但更兼容）
    # model_max_length: 截断超长序列至 8192 个 token
    # -----------------------------------------------------------------------
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )

    # -----------------------------------------------------------------------
    # 【阶段6b】参数冻结/解冻设置
    # 两种模式二选一：LoRA 低秩适配 或 全参数精细控制
    # -----------------------------------------------------------------------
    if training_args.lora_enable:
        # LoRA 模式：冻结所有参数，只插入并训练低秩矩阵
        from peft import LoraConfig, get_peft_model, TaskType
        print("LoRA enabled")

        for p in model.parameters():
            p.requires_grad = False  # 先冻结全部参数

        lora_config = LoraConfig(
            r=training_args.lora_r or 64,                    # 低秩维度
            lora_alpha=training_args.lora_alpha or 128,       # 缩放因子
            lora_dropout=training_args.lora_dropout or 0.05,  # Dropout 比例
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # Qwen 的 attention 线性层
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        model = get_peft_model(model, lora_config)  # 插入 LoRA 适配器
    else:
        # 全参数模式：根据 tune_mm_* 标志精细控制各模块是否可训练
        set_model(model_args, model)

        # 仅在主进程（rank 0）打印可训练参数信息，避免重复输出
        if torch.distributed.get_rank() == 0:
            model.visual.print_trainable_parameters()    # ViT 可训练状态（monkey patch 方法）
            model.model.print_trainable_parameters()     # LLM 可训练状态（monkey patch 方法）
    
    # -----------------------------------------------------------------------
    # 【阶段7】构建数据模块
    # make_supervised_data_module 返回 dict，包含：
    #   train_dataset:  LazySupervisedDataset 实例（懒加载，__getitem__ 时才处理）
    #   eval_dataset:   None（sft_qwen3_4b.sh 中 eval_strategy="no"）
    #   data_collator:  FlattenedDataCollatorForSupervisedDataset 或
    #                   DataCollatorForSupervisedDataset
    # -----------------------------------------------------------------------
    data_module = make_supervised_data_module(processor, data_args=data_args)

    # -----------------------------------------------------------------------
    # 【阶段8】初始化 HuggingFace Trainer 并启动训练
    # Trainer 的 create_optimizer 已在 trainer.py 中被 monkey patch，
    # 支持 mm_projector_lr 和 vision_tower_lr 独立学习率
    # DeepSpeed 通过 training_args.deepspeed 自动集成到 Trainer
    # -----------------------------------------------------------------------
    trainer = Trainer(
        model=model, processing_class=tokenizer, args=training_args, **data_module
    )

    # 检测 output_dir 是否已有 checkpoint（支持断点续训）
    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        logging.info("checkpoint found, resume training")
        trainer.train(resume_from_checkpoint=True)  # 从最新 checkpoint 恢复训练
    else:
        trainer.train()  # 从头开始训练

    # 保存 Trainer 内部状态（如 global_step、训练日志等）
    trainer.save_state()

    # -----------------------------------------------------------------------
    # 【阶段9】保存最终模型
    # 先恢复 use_cache=True（推理时需要 KV Cache）
    # 再调用 safe_save_model_for_hf_trainer 处理 ZeRO-3 分片聚合问题
    # 最后保存 processor（包含 tokenizer 配置和 image_processor 配置）
    # -----------------------------------------------------------------------
    model.config.use_cache = True

    safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)
    
    # 保存 processor（image_processor + tokenizer 配置），推理时需要一起加载
    processor.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    # 程序入口：torchrun 在每个 GPU 上启动一个独立进程，各自调用 train()
    # FlashAttention-2 需要安装 flash-attn 库并拥有 Ampere+ GPU（A100/H100 等）
    train(attn_implementation="flash_attention_2")
