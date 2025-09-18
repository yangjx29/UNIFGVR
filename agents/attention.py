# qwen_attention_intervention.py
import math
import types
from typing import Optional, Tuple, Callable

import torch
import torch.nn as nn

# 注意：确保你的环境中能 import 下列函数（与 Qwen2.5-VL 源码一致）
# from qwen.modeling import apply_multimodal_rotary_pos_emb  # adjust import path if needed
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import apply_multimodal_rotary_pos_emb, repeat_kv 


def qwen_new_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[object] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    cache_position: Optional[torch.LongTensor] = None,
    position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
    **kwargs,
) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[tuple[torch.Tensor]]]:
    """
    替换后的 forward：手工计算 attention logits，以便在 softmax 之前修改 logits（放大 image token 注意力）。
    返回 (attn_output, attn_weights, past_key_values)
    """
    bsz, q_len, _ = hidden_states.size()

    # Q/K/V 投影（与原实现一致）
    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)

    # reshape -> (bsz, num_heads, seq, head_dim)
    query_states = query_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)

    # get rotary embeddings (multimodal)
    if position_embeddings is None:
        # 保守处理：若 position_embeddings 未传入，尝试从 self 或 kwargs 获取（某些实现会如此）
        position_embeddings = getattr(self, "position_embeddings", None) or kwargs.get("position_embeddings", None)

    if position_embeddings is None:
        # 如果仍然没有，跳过 RoPE（不推荐，但可防止崩溃）
        cos = sin = None
        print(f'cos = sin = None')
    else:
        cos, sin = position_embeddings
        # 进行位置编码
        query_states, key_states = apply_multimodal_rotary_pos_emb(
            query_states, key_states, cos, sin, getattr(self, "rope_scaling", {}).get("mrope_section", None)
        )

    # past_key_values 更新（与原实现一致）
    kv_seq_len = key_states.shape[-2]  # seq len for keys/values
    if past_key_values is not None:
        if getattr(self, "layer_idx", None) is None:
            raise ValueError(
                "If using k/v caching with this attention, ensure attention layer has attribute `layer_idx`."
            )
        # 如果 past_key_values 提供了 update 接口（与 qwen 实现一致）
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)
        # kv_seq_len 可能已增长，按实际 shape 重新确定
        kv_seq_len = key_states.shape[-2]

    # ---------- 手工计算 attention logits（以便修改） ----------
    # shape: (bsz, num_heads, q_len, kv_seq_len)
    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)
    attn_logits = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
    
    # 校验形状（与原实现的期望一致）
    expected_shape = (bsz, self.num_heads, q_len, kv_seq_len)
    if attn_logits.size() != expected_shape:
        # 尝试容错：如果 key/value 的 num_heads 压缩（比如 num_key_value_heads < num_attention_heads），
        # 有些实现会把 key/value 按 group 投影。这里我们不做复杂 group 处理，直接报错以便用户注意。
        raise ValueError(f"Unexpected attn_logits shape {attn_logits.size()}, expected {expected_shape}.")

    # attention_mask 应为 (bsz, 1, q_len, kv_seq_len)（与 Qwen 源码一致）
    if attention_mask is not None:
        print(f'attention_mask is not None测试')
        if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
            raise ValueError(
                f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
            )
        # TODO 掩码的作用调研
        attn_logits = attn_logits + attention_mask
    # 避免数值下溢导致 NaN
    attn_logits = torch.max(attn_logits, torch.tensor(torch.finfo(attn_logits.dtype).min, device=attn_logits.device))
    # if torch.isnan(attn_logits).any() or torch.isinf(attn_logits).any():
    #     print("NaN/Inf detected in attn_logits 测试2 max之后:")

    # ========== PAI-like modification: 放大 image token 的 logits（仅作用于当前生成 token） ==========
    # 兼容性：检查 attention module 上的控制开关
    use_attn = getattr(self, "use_attn", False)
    use_cfg = getattr(self, "use_cfg", False)
    alpha = getattr(self, "alpha", 0.0)
    img_start_idx = getattr(self, "img_start_idx", None)
    img_end_idx = getattr(self, "img_end_idx", None)

    # 只在启用 use_attn 且未启用 use_cfg 时直接修改 logits（与 PAI 一致）
    if use_attn and (not use_cfg) and (img_start_idx is not None) and (img_end_idx is not None):
        # 只修改查询序列的最后一个 token 对图像 token 的 logits（自回归生成场景）
        # attn_logits[:, :, -1, img_start_idx:img_end_idx] = |w|*alpha + w
        target = attn_logits[:, :, -1, img_start_idx:img_end_idx]
        # 保持 device & dtype
        attn_logits[:, :, -1, img_start_idx:img_end_idx] = target.abs() * alpha + target
    else:
        print(f'未进行修改')
    # softmax -> attn_probs
    if torch.isnan(attn_logits).any() or torch.isinf(attn_logits).any():
        print("NaN/Inf detected in attn_logits222测试")
    

    attn_probs = nn.functional.softmax(attn_logits, dim=-1, dtype=torch.float32).to(query_states.dtype)
    if torch.isnan(attn_probs).any() or torch.isinf(attn_probs).any():
        print("NaN/Inf detected in attn_probs:")
    # attn output = attn_probs @ V
    attn_output = torch.matmul(attn_probs, value_states)  # (bsz, num_heads, q_len, head_dim)

    # 校验输出形状
    if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
        raise ValueError(
            f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is {attn_output.size()}"
        )

    # 恢复维度 (bsz, q_len, hidden)
    attn_output = attn_output.transpose(1, 2).reshape(bsz, q_len, -1).contiguous()
    attn_output = self.o_proj(attn_output)

    # 根据 output_attentions 决定是否返回 attn_probs（注意：返回的 attn_probs dtype 与原实现可能不同）
    attn_weights = attn_probs if output_attentions else None

    # 根据 use_cache 决定是否要返回缓存（这里直接返回原 past_key_values）
    return attn_output, attn_weights


def qwen_modify(model, start_layer: int, end_layer: int, use_attn: bool, alpha: float, use_cfg: bool,
                img_start_idx: int, img_end_idx: int):
    """
    在 model 指定层范围内注入注意力干预参数并替换 forward。
    尝试在常见 attention 属性名上注入（self_attn, attn, attention）。
    """
    for i in range(start_layer, end_layer):
        attn = model.model.language_model.layers[i].self_attn
        attn.use_attn = use_attn
        attn.alpha = alpha
        attn.use_cfg = use_cfg
        attn.img_start_idx = img_start_idx
        attn.img_end_idx = img_end_idx
        attn.layer_idx = i
        attn.forward = types.MethodType(qwen_new_forward, attn)
    # 返回模型以便链式调用
    return model
