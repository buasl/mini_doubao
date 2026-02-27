# =============================================================================
# trainer.py —— 自定义 Trainer 扩展与 Monkey Patch 模块
#
# 本文件主要完成以下三件事：
#
# 1. 变长 FlashAttention 实现（flash_attention_forward / qwen2vl_forward / qwen3vl_forward）
#    —— 当启用 data_flatten 时，多个样本被拼接成一条长序列，
#       需要用 cu_seqlens（各样本长度的前缀和）区分样本边界再计算注意力
#
# 2. replace_qwen2_vl_attention_class()
#    —— Monkey Patch：将 Qwen2/2.5/3 VL 各系列的 Attention.forward 替换为左边自定义版本
#
# 3. create_optimizer（替换 Trainer.create_optimizer）
#    —— 支持为 MLP Merger、ViT、LLM 主干设置不同学习率（可选）
#
# 4. 可训练参数打印方法（添加到各 VisionModel 和 VLModel 上）
# =============================================================================

from typing import Dict, List, Optional, Sequence, Tuple, Callable

import torch
# flash_attn_varlen_func: FlashAttention 的变长序列接口，输入 cu_seqlens 而非 padding mask
from flash_attn.flash_attn_interface import flash_attn_varlen_func
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers import Trainer
from transformers.cache_utils import Cache
from transformers.utils.deprecation import deprecate_kwarg
from transformers.processing_utils import Unpack
from transformers.models.qwen2_vl.modeling_qwen2_vl import (
    Qwen2VisionTransformerPretrainedModel,
    Qwen2VLModel,
    apply_multimodal_rotary_pos_emb,  # Qwen2-VL 专用的多模态 RoPE 应用函数
)
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
    Qwen2_5_VisionTransformerPretrainedModel,
    Qwen2_5_VLModel,
)
from transformers.models.qwen3_vl.modeling_qwen3_vl import (
    Qwen3VLVisionModel,
    Qwen3VLModel,
    apply_rotary_pos_emb,  # Qwen3-VL 的标准 RoPE 应用函数
)
from transformers.models.qwen3_vl_moe.modeling_qwen3_vl_moe import (
    Qwen3VLMoeVisionModel,
    Qwen3VLMoeModel,
)
from transformers.utils import logging

logger = logging.get_logger(__name__)


def flash_attention_forward(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    dropout: float = 0.0,
    scaling: Optional[float] = None,
    sliding_window: Optional[int] = None,
    softcap: Optional[float] = None,
    **kwargs,
) -> tuple[torch.Tensor, None]:
    """变长序列 FlashAttention 底层实现。

    核心差异：这里的 attention_mask 不是普通的 0/1 掉盘 mask，
    而是 cu_seqlens（cumulative sequence lengths，各样本长度的前缀和张量）。
    例： batch中有 3 个样本，长度分别为 10, 15, 20，则 cu_seqlens=[0,10,25,45]。
    flash_attn_varlen_func 能根据 cu_seqlens 正确地将注意力限制在各自样本内部。
    """
    if kwargs.get("output_attentions", False) or kwargs.get("head_mask") is not None:
        logger.warning_once(
            "`flash_attention_2` does not support `output_attentions=True` or `head_mask`."
            " Please set your attention to `eager` if you want any of these features."
        )
    
    # This is before the transpose
    seq_len = query.shape[2]

    if any(dim == 0 for dim in query.shape):
        raise ValueError(
            "Tensor query has shape  with a zero dimension.\n"
            "FlashAttention does not support inputs with dim=0.\n"
            "Please check your input shapes or use SDPA instead."
        )
    # FA2 小要求输入格式为 [batch, seqlen, head, dim]（非转置状态）
    # 当前 q/k/v 格式为 [batch, head, seq_len, dim]，需转置
    # batch, head, seq_len, dim
    query = query.transpose(1, 2)
    key = key.transpose(1, 2)
    value = value.transpose(1, 2)
    # batch, seqlen, head, dim

    # In PEFT, usually we cast the layer norms in float32 for training stability reasons
    # therefore the input hidden states gets silently casted in float32. Hence, we need
    # cast them back in the correct dtype just to be sure everything works as expected.
    # This might slowdown training & inference so it is recommended to not cast the LayerNorms
    # in fp32. (usually our RMSNorm modules handle it correctly)
    # 当输入被静默升级为 float32时，需要拉回模型实际精度
    target_dtype = None
    if query.dtype == torch.float32:
        if torch.is_autocast_enabled():
            target_dtype = torch.get_autocast_gpu_dtype()
        # Handle the case where the model is quantized
        elif hasattr(module.config, "_pre_quantization_dtype"):
            target_dtype = module.config._pre_quantization_dtype
        else:
            target_dtype = next(layer for layer in module.modules() if isinstance(layer, torch.nn.Linear)).weight.dtype

    # 去掉 batch 维度（已拼接成一条长序列，不再需要 batch 维）
    query = query.squeeze(0)
    key = key.squeeze(0)
    value = value.squeeze(0)
    # attention_mask 在 data_flatten 模式下传入的是 cu_seqlens
    cu_seqlens = attention_mask

    # 计算 batch 中最长序列的长度（flash_attn_varlen_func 需要此参数优化内存）
    with torch.no_grad():
        max_seqlen = max(
            [
                cu_seqlens[idx + 1] - cu_seqlens[idx]
                for idx in range(cu_seqlens.size(0) - 1)
            ]
        ).item()

    # 调用 FlashAttention 变长序列接口，causal=True 表示因果注意力（下三角 mask）
    attn_output = flash_attn_varlen_func(
        query,
        key,
        value,
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_k=cu_seqlens,
        max_seqlen_q=max_seqlen,
        max_seqlen_k=max_seqlen,
        causal=True,
    )

    # 恢复 batch 维度以匹配下游层的期望形状
    attn_output = attn_output.unsqueeze(0)

    return attn_output, None


@deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
def qwen2vl_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[Cache] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    cache_position: Optional[torch.LongTensor] = None,
    position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,  # necessary, but kept here for BC
    **kwargs: Unpack[FlashAttentionKwargs],
) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[tuple[torch.Tensor]]]:
    """替换 Qwen2-VL 和 Qwen2.5-VL 的 Attention.forward 方法。

    与官方实现的差异：
      - 直接调用本文件的 flash_attention_forward（支持 cu_seqlens 变长注意力）
      - attention_mask 此时传递的是 cu_seqlens 而非常规掩码
    """
    bsz, q_len, _ = hidden_states.size()

    # 线性投影：将隐层状态投影为 Q/K/V
    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)

    # 重形为 [batch, head, seq_len, head_dim] 以方便 RoPE 应用
    query_states = query_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)

    # 应用多模态 RoPE 位置编码（分 time/height/width 三个维度，对应 mrope_section 分割）
    cos, sin = position_embeddings
    query_states, key_states = apply_multimodal_rotary_pos_emb(
        query_states, key_states, cos, sin, self.rope_scaling["mrope_section"]
    )

    if past_key_values is not None:
        # 更新 KV 缓存（训练时 use_cache=False，此分支不会执行）
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}  # Specific to RoPE models
        key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)

    # 调用自定义的变长 FlashAttention，传入 cu_seqlens 作为 attention_mask
    attn_output, attn_weights = flash_attention_forward(
        self,
        query_states,
        key_states,
        value_states,
        attention_mask,
        dropout=0.0 if not self.training else self.attention_dropout,
        scaling=self.scaling,
        sliding_window=self.sliding_window,
        position_ids=position_ids,  # pass positions for FA2
        **kwargs,
    )

    # 将输出重形为 [batch, seq_len, hidden_size]，再经过输出投影
    attn_output = attn_output.reshape(bsz, q_len, -1).contiguous()
    attn_output = self.o_proj(attn_output)
    return attn_output, attn_weights



@deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
def qwen3vl_forward(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    attention_mask: Optional[torch.Tensor],
    past_key_values: Optional[Cache] = None,
    cache_position: Optional[torch.LongTensor] = None,
    **kwargs: Unpack[FlashAttentionKwargs],
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    """替换 Qwen3-VL 和 Qwen3-VL MoE 的 Attention.forward 方法。

    Qwen3-VL 与 Qwen2-VL 的差异：
      - Q/K 头部应用了 QK-Norm（q_norm / k_norm），提高训练稳定性
      - 使用标准 RoPE （apply_rotary_pos_emb）而非多模态分段 RoPE
    """
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)

    # 投影 + 重形 + QK-Norm（Qwen3-VL 新增的归一化操作）+ 转置
    query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)  # V 无需 norm

    # 应用标准 RoPE 位置编码
    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    if past_key_values is not None:
        # sin and cos are specific to RoPE models; cache_position needed for the static cache
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)

    # 调用自定义变长 FlashAttention
    attn_output, attn_weights = flash_attention_forward(
        self,
        query_states,
        key_states,
        value_states,
        attention_mask,
        dropout=0.0 if not self.training else self.attention_dropout,
        scaling=self.scaling,
        **kwargs,
    )

    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    attn_output = self.o_proj(attn_output)
    return attn_output, attn_weights


def return_mask(
    config,
    input_embeds,
    attention_mask,
    cache_position,
    past_key_values,
    position_ids,
    **kwargs
):
    """直接返回传入的 attention_mask，不做任何变换。

    这个函数用来替换 create_causal_mask / create_sliding_window_causal_mask。
    在 data_flatten 模式下，我们的 attention_mask 已是 cu_seqlens，
    不需要官方逻辑再构建四维 [bsz, 1, q_len, kv_len] 的 mask。
    """
    return attention_mask


def replace_qwen2_vl_attention_class():
    """执行 Monkey Patch：将四种 Qwen VL 模型的 Attention 层替换为变长序列实现。

    替换内容：
      1. Attention.forward → 自定义的 qwen2vl_forward / qwen3vl_forward
         （支持传入 cu_seqlens 的变长 FlashAttention）
      2. create_causal_mask → return_mask
         （不生成完整 4D 因果 mask，直接返回 cu_seqlens）
      3. create_sliding_window_causal_mask → return_mask （同上）

    这样 Trainer 中的训练循环调用 model.forward() 时，
    会自动使用替换后的变长实现，实现高效的 packed sequence 训练。
    """
    import transformers
    import transformers.modeling_flash_attention_utils

    # --- Qwen2-VL ---
    transformers.models.qwen2_vl.modeling_qwen2_vl.Qwen2VLAttention.forward = (
        qwen2vl_forward
    )
    transformers.models.qwen2_vl.modeling_qwen2_vl.create_causal_mask = (
        return_mask
    )
    transformers.models.qwen2_vl.modeling_qwen2_vl.create_sliding_window_causal_mask = (
        return_mask
    )    
    ## qwen2_5_vl
    # --- Qwen2.5-VL ---
    transformers.models.qwen2_5_vl.modeling_qwen2_5_vl.Qwen2_5_VLAttention.forward = (
        qwen2vl_forward
    )
    transformers.models.qwen2_5_vl.modeling_qwen2_5_vl.create_causal_mask = (
        return_mask
    )
    transformers.models.qwen2_5_vl.modeling_qwen2_5_vl.create_sliding_window_causal_mask = (
        return_mask
    )
    ## qwen3vl
    # --- Qwen3-VL 密集模型 ---
    transformers.models.qwen3_vl.modeling_qwen3_vl.Qwen3VLTextAttention.forward = (
        qwen3vl_forward
    )
    transformers.models.qwen3_vl.modeling_qwen3_vl.create_causal_mask = (
        return_mask
    )
    ## qwen3vl moe
    # --- Qwen3-VL MoE 模型 ---
    transformers.models.qwen3_vl_moe.modeling_qwen3_vl_moe.Qwen3VLMoeTextAttention.forward = (
        qwen3vl_forward
    )
    transformers.models.qwen3_vl_moe.modeling_qwen3_vl_moe.create_causal_mask = (
        return_mask
    )


def print_trainable_parameters_visual(self) -> None:
    """
    Prints the trainable status of all vision components including attention blocks and merger module.
    Outputs the indices of trainable/non-trainable blocks and the merger module status.
    """
    trainable_blocks = []
    non_trainable_blocks = []

    # Check trainable status of vision attention blocks
    for block_idx, block in enumerate(self.blocks):
        is_trainable = all(param.requires_grad for param in block.parameters())
        if is_trainable:
            trainable_blocks.append(block_idx)
        else:
            non_trainable_blocks.append(block_idx)

    # Check trainable status of merger module
    is_merger_trainable = any(param.requires_grad for param in self.merger.parameters())

    # Print results
    print("Vision Module - Attention Blocks:")
    print(
        f"Trainable Block Indices: {trainable_blocks if trainable_blocks else 'None'}"
    )
    print(
        f"Non-Trainable Block Indices: {non_trainable_blocks if non_trainable_blocks else 'None'}"
    )
    print(f"Merger Module Trainable: {is_merger_trainable}")


def print_trainable_parameters(self) -> None:
    """
    Prints the trainable status of all LLM components including embeddings, layers, and normalization.
    Outputs the indices of trainable/non-trainable layers and other module statuses.
    """
    # Check embed_tokens
    is_embed_trainable = any(
        param.requires_grad for param in self.language_model.embed_tokens.parameters()
    )
    print(f"LLM Module - Embed Tokens Trainable: {is_embed_trainable}")

    # Check each decoder layer
    trainable_layers = []
    non_trainable_layers = []

    for layer_idx, layer in enumerate(self.language_model.layers):
        is_trainable = any(param.requires_grad for param in layer.parameters())
        if is_trainable:
            trainable_layers.append(layer_idx)
        else:
            non_trainable_layers.append(layer_idx)

    # Print layer status
    print(
        f"LLM Module - Trainable Layer Indices: {trainable_layers if trainable_layers else 'None'}"
    )
    print(
        f"LLM Module - Non-Trainable Layer Indices: {non_trainable_layers if non_trainable_layers else 'None'}"
    )


def create_optimizer(self):
    """自定义优化器创建逻辑，替换 Trainer 默认实现。

    扩展功能：支持为不同组件设置独立学习率（通过 TrainingArguments 中的对应字段指定）：
      - mm_projector_lr: MLP Merger 投影层的学习率
      - vision_tower_lr: ViT 视觉编码器的学习率
      - 其余参数使用全局 learning_rate

    参数分组逻辑：
      1. 若 mm_projector_lr 且 vision_tower_lr 均指定 → 6 个参数组
      2. 若仅指定 mm_projector_lr            → 4 个参数组
      3. 均未指定                            → 2 个参数组（标准 weight decay 分组）

    每个参数组可独立设置 lr，其他参数（weight_decay等）随全局默认。
    """
    opt_model = self.model

    if self.optimizer is None:
        # 获取需要 weight decay 的参数名称列表（通常是非 bias 的权重参数）
        decay_parameters = self.get_decay_parameter_names(opt_model)
        decay_parameters = [name for name in decay_parameters if "bias" not in name]
        if self.args.mm_projector_lr is not None and self.args.mm_projector_lr != 0:
            projector_parameters = [
                name for name, _ in opt_model.named_parameters() if "merger" in name
            ]
            if self.args.vision_tower_lr is not None and self.args.vision_tower_lr != 0:
                vision_tower_parameters = [
                    name for name, _ in opt_model.named_parameters() if "visual" in name
                ]
                optimizer_grouped_parameters = [
                    {
                        "params": [
                            p
                            for n, p in opt_model.named_parameters()
                            if (
                                n in decay_parameters
                                and n not in projector_parameters
                                and n not in vision_tower_parameters
                                and p.requires_grad
                            )
                        ],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [
                            p
                            for n, p in opt_model.named_parameters()
                            if (
                                n in decay_parameters
                                and n not in projector_parameters
                                and n in vision_tower_parameters
                                and p.requires_grad
                            )
                        ],
                        "weight_decay": self.args.weight_decay,
                        "lr": self.args.vision_tower_lr,
                    },
                    {
                        "params": [
                            p
                            for n, p in opt_model.named_parameters()
                            if (
                                n not in decay_parameters
                                and n not in projector_parameters
                                and n not in vision_tower_parameters
                                and p.requires_grad
                            )
                        ],
                        "weight_decay": 0.0,
                    },
                    {
                        "params": [
                            p
                            for n, p in opt_model.named_parameters()
                            if (
                                n not in decay_parameters
                                and n not in projector_parameters
                                and n in vision_tower_parameters
                                and p.requires_grad
                            )
                        ],
                        "weight_decay": 0.0,
                        "lr": self.args.vision_tower_lr,
                    },
                    {
                        "params": [
                            p
                            for n, p in opt_model.named_parameters()
                            if (
                                n in decay_parameters
                                and n in projector_parameters
                                and p.requires_grad
                            )
                        ],
                        "weight_decay": self.args.weight_decay,
                        "lr": self.args.mm_projector_lr,
                    },
                    {
                        "params": [
                            p
                            for n, p in opt_model.named_parameters()
                            if (
                                n not in decay_parameters
                                and n in projector_parameters
                                and p.requires_grad
                            )
                        ],
                        "weight_decay": 0.0,
                        "lr": self.args.mm_projector_lr,
                    },
                ]
            else:
                optimizer_grouped_parameters = [
                    {
                        "params": [
                            p
                            for n, p in opt_model.named_parameters()
                            if (
                                n in decay_parameters
                                and n not in projector_parameters
                                and p.requires_grad
                            )
                        ],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [
                            p
                            for n, p in opt_model.named_parameters()
                            if (
                                n not in decay_parameters
                                and n not in projector_parameters
                                and p.requires_grad
                            )
                        ],
                        "weight_decay": 0.0,
                    },
                    {
                        "params": [
                            p
                            for n, p in opt_model.named_parameters()
                            if (
                                n in decay_parameters
                                and n in projector_parameters
                                and p.requires_grad
                            )
                        ],
                        "weight_decay": self.args.weight_decay,
                        "lr": self.args.mm_projector_lr,
                    },
                    {
                        "params": [
                            p
                            for n, p in opt_model.named_parameters()
                            if (
                                n not in decay_parameters
                                and n in projector_parameters
                                and p.requires_grad
                            )
                        ],
                        "weight_decay": 0.0,
                        "lr": self.args.mm_projector_lr,
                    },
                ]
        else:
            optimizer_grouped_parameters = [
                {
                    "params": [
                        p
                        for n, p in opt_model.named_parameters()
                        if (n in decay_parameters and p.requires_grad)
                    ],
                    "weight_decay": self.args.weight_decay,
                },
                {
                    "params": [
                        p
                        for n, p in opt_model.named_parameters()
                        if (n not in decay_parameters and p.requires_grad)
                    ],
                    "weight_decay": 0.0,
                },
            ]

        # 根据 Training args 获取优化器类和默认 kwargs（如 lr、betas等）
        optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(
            self.args
        )
        # 用分组参数初始化优化器（不同组可有不同的 lr / weight_decay）
        self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)

    return self.optimizer


# =============================================================================
# 文件加载时立即执行的 Monkey Patch 赋值操作
# 这些操作在 import trainer 模块时就会生效，修改全局类实例的方法
# =============================================================================

# 替换 HuggingFace Trainer 的优化器创建逻辑，支持多学习率分组
Trainer.create_optimizer = create_optimizer

# 将可训练参数打印方法挂载到各类型的 ViT 视觉模型和 VL 主干上
Qwen2VisionTransformerPretrainedModel.print_trainable_parameters = (
    print_trainable_parameters_visual
)
Qwen2VLModel.print_trainable_parameters = print_trainable_parameters
Qwen2_5_VisionTransformerPretrainedModel.print_trainable_parameters = (
    print_trainable_parameters_visual
)
Qwen2_5_VLModel.print_trainable_parameters = print_trainable_parameters

Qwen3VLVisionModel.print_trainable_parameters = (
    print_trainable_parameters_visual
)
Qwen3VLModel.print_trainable_parameters = print_trainable_parameters
Qwen3VLMoeVisionModel.print_trainable_parameters = print_trainable_parameters_visual
Qwen3VLMoeModel.print_trainable_parameters = print_trainable_parameters