import transformers
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, List

# =============================================================================
# 训练参数定义模块
# 本文件定义三组 dataclass 参数，由 HfArgumentParser 解析命令行并输入给 train_qwen.py
# =============================================================================


@dataclass
class ModelArguments:
    """模型架构相关参数：控制加载哪个模型以及哪些组件参与训练"""
    # 预训练模型路径或 HuggingFace Hub Model ID
    model_name_or_path: Optional[str] = field(default="Qwen/Qwen2.5-VL-3B-Instruct")
    # 是否训练 LLM 语言模型主干（Transformer decoder 层）
    tune_mm_llm: bool = field(default=False)
    # 是否训练视觉-语言连接 MLP Merger（将视觉 token 投影到 LLM 空间的投影层）
    tune_mm_mlp: bool = field(default=False)
    # 是否训练视觉编码器 ViT（Qwen-VL 中的 visual transformer 块）
    tune_mm_vision: bool = field(default=False)

@dataclass
class DataArguments:
    """数据加载与预处理相关参数"""
    # 数据集名称，逗号分隔，对应 qwenvl/data/__init__.py 中 data_dict 的键
    dataset_use: str = field(default="")
    # 验证集名称，对应 data_dict 的键，用于 epoch 级别的 eval
    eval_dataset_use: Optional[str] = field(default=None)
    # 是否启用 sequence packing（将多个样本拼接成长序列）
    # 启用后使用 FlattenedDataCollator 和变长 FlashAttention，减少 padding
    data_flatten: bool = field(default=False)
    # 是否启用 data packing（与 data_flatten 类似）
    data_packing: bool = field(default=False)
    # 基础采样间隔（用于视频帧采样）
    base_interval: int = field(default=2)
    # 图像处理时允许的最大像素数（控制图像被缩放的上限）
    max_pixels: int = field(default=28 * 28 * 576)
    # 图像处理时允许的最小像素数（控制图像被放大的下限）
    min_pixels: int = field(default=28 * 28 * 16)
    # 视频采样的最大/最小帧数
    video_max_frames: Optional[int] = field(default=8)
    video_min_frames: Optional[int] = field(default=4)
    # 视频每帧允许的最大/最小像素数
    video_max_pixels: int = field(default=1024 * 28 * 28)
    video_min_pixels: int = field(default=256 * 28 * 28)
    # 视频帧率（每秒采样帧数）
    video_fps: float = 2
    # 注意：model_type 不在此处定义，由 train_qwen.py 在解析参数后动态赋值
    # 取值： "qwen3vl" / "qwen2.5vl" / "qwen2vl"


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    """训练过程控制参数，继承自 HuggingFace TrainingArguments，扩展了多模态微调和 LoRA 选项"""
    # HuggingFace 模型缓存目录，保存下载的预训练权重
    cache_dir: Optional[str] = field(default=None)
    # 优化器类型，默认使用 PyTorch 内置的 AdamW
    optim: str = field(default="adamw_torch")
    # 最大输入序列长度，超出则截断（右填冲）
    model_max_length: int = field(
        default=512,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    # MLP Merger 投影层的独立学习率（None 表示与全局 lr 相同）
    mm_projector_lr: Optional[float] = None
    # ViT 视觉编码器的独立学习率（None 表示与全局 lr 相同）
    vision_tower_lr: Optional[float] = None

    ## Lora config
    # 是否启用 LoRA（低秩适配）微调模式——冻结原始参数，只训练插入的小矩阵
    lora_enable: bool = field(default=False)
    # LoRA 秩（内秘 dimension），改变参数数量的关键超参数
    lora_r: int = field(default=64)
    # LoRA 缩放因子，通常设为 2r–4r
    lora_alpha: int = field(default=128)
    # LoRA dropout 比例，防止过拟合
    lora_dropout: float = field(default=0.0)
