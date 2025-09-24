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
        # print(f'attention_mask is not None测试')
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
    
    # 注意力增强：仅对图像token区域进行增强
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


def qwen_new_forward_with_importance(
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
    替换后的 forward：使用注意力重要性权重进行差异化增强。
    返回 (attn_output, attn_weights, past_key_values)
    """
    # 只有初始的q_len长度大于1,之后都只会保留生成的最后一个token
    bsz, q_len, _ = hidden_states.size()
    print(f"hidden_states:{hidden_states.shape}")
    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)

    # reshape -> (bsz, num_heads, seq, head_dim)
    query_states = query_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
    
    cos, sin = position_embeddings
    # 进行位置编码
    query_states, key_states = apply_multimodal_rotary_pos_emb(
        query_states, key_states, cos, sin, self.rope_scaling["mrope_section"]
    )

    kv_seq_len = key_states.shape[-2]  # seq len for keys/values
    if past_key_values is not None:
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position} # Specific to RoPE models
        key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)
        kv_seq_len = key_states.shape[-2]

    # shape: (bsz, num_heads, q_len, kv_seq_len)
    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)

    # attention_mask 应为 (bsz, 1, q_len, kv_seq_len)
    causal_mask = attention_mask
    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]


    if query_states.device.type == "cuda" and attention_mask is not None:
        query_states = query_states.contiguous()
        key_states = key_states.contiguous()
        value_states = value_states.contiguous()

    is_causal = True if causal_mask is None and q_len > 1 else False
    # =========== scaled_dot_product_attention源码============
    scale=None
    enable_gqa=False
    dropout_p=self.attention_dropout if self.training else 0.0
    
    L, S = query_states.size(-2), key_states.size(-2)
    scale_factor = 1 / math.sqrt(query_states.size(-1)) if scale is None else scale
    attn_bias = torch.zeros(L, S, dtype=query_states.dtype, device=query_states.device)
    if is_causal:
        assert causal_mask is None
        device = query_states.device
        temp_mask = torch.ones(L, S, dtype=torch.bool, device=device).tril(diagonal=0)
        attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
        attn_bias.to(query_states.dtype)
    if causal_mask is not None:
        if causal_mask.dtype == torch.bool:
            attn_bias.masked_fill_(causal_mask.logical_not(), float("-inf"))
        else:
            attn_bias = causal_mask + attn_bias
    if enable_gqa:
        key_states = key_states.repeat_interleave(query_states.size(-3)//key_states.size(-3), -3)
        value = value.repeat_interleave(query_states.size(-3)//value.size(-3), -3)
    attn_weight = query_states @ key_states.transpose(-2, -1) * scale_factor
    attn_weight += attn_bias
    attn_weight = torch.softmax(attn_weight, dim=-1)
    attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
    # 进行增强
    # ==========根据重要性权重差异化增强 image token 的 logits ==========
    # use_attn = getattr(self, "use_attn", False)
    # use_cfg = getattr(self, "use_cfg", False)
    # alpha = getattr(self, "alpha", 0.0)
    # img_start_idx = getattr(self, "img_start_idx", None)
    # img_end_idx = getattr(self, "img_end_idx", None)
    # importance_weights = getattr(self, "importance_weights", None)
    # important_indices = getattr(self, "important_indices", None)
    # if use_attn and importance_weights is not None:
    #     device = attn_weight.device
    #     if importance_weights.device != device:
    #         importance_weights = importance_weights.to(device)
    #     kv_pos_begin = img_start_idx + 1
    #     kv_pos_end = img_end_idx - 1
    #     # 检查切片是否有效
    #     if kv_pos_begin < kv_pos_end and kv_pos_end <= attn_weight.shape[-1]:
    #         # 只处理当前查询 token（通常是最后一个）对图像 token 的注意力
    #         target_weight = attn_weight[:, :, -1, kv_pos_begin:kv_pos_end]

    #         # 确保重要性权重的长度与目标注意力张量匹配
    #         num_image_tokens = target_weight.shape[-1]
    #         if len(importance_weights) != num_image_tokens:
    #             if len(importance_weights) > num_image_tokens:
    #                 importance_weights = importance_weights[:num_image_tokens]
    #             else:
    #                 padding = torch.zeros(num_image_tokens - len(importance_weights), device=device, dtype=importance_weights.dtype)
    #                 importance_weights = torch.cat([importance_weights, padding])

    #         # 创建增强掩码
    #         enhancement_threshold = 1.0
    #         enhancement_mask = importance_weights > enhancement_threshold
            
    #         if enhancement_mask.any():
    #             # 广播掩码和权重以匹配注意力张量的形状
    #             enhancement_mask_broadcast = enhancement_mask.view(1, 1, -1)
                
    #             # 增强逻辑
    #             enhancement = target_weight.abs() * alpha
    #             enhanced_logits = target_weight + enhancement

    #             # 应用增强
    #             final_enhanced_weight = torch.where(
    #                 enhancement_mask_broadcast,
    #                 enhanced_logits,
    #                 target_weight
    #             )

    #             # 将增强后的权重写回到原始的 attn_weight 张量中
    #             attn_weight[:, :, -1, kv_pos_begin:kv_pos_end] = final_enhanced_weight
    #         else:
    #             print("[ATTN] No tokens meet enhancement threshold.")
    #     else:
    #         print(f"[ATTN] Invalid image token slice range. kv_pos_begin:{kv_pos_begin},kv_pos_end:{kv_pos_end} attn_weight.shape[-1]:{attn_weight.shape[-1]}")
    # attn_output = attn_weight @ value_states
    # =======源码结束
    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.view(bsz, q_len, self.hidden_size)

    attn_output = self.o_proj(attn_output)
    return attn_output, None, past_key_values
    # attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
    # print(f'attn_weights after matmul:{attn_weights.shape}')
    # # 校验形状（与原实现的期望一致）
    # expected_shape = (bsz, self.num_heads, q_len, kv_seq_len)
    # if attn_weights.size() != expected_shape:
    #     # 有些实现会把 key/value 按 group 投影。这里我们不做复杂 group 处理，直接报错以便用户注意。
    #     raise ValueError(f"Unexpected attn_logits shape {attn_weights.size()}, expected {expected_shape}.")


    # if query_states.dtype == torch.float16:
    #     attn_weights = torch.where(torch.isinf(attn_weights), torch.zeros_like(attn_weights), attn_weights)

    # ==========根据重要性权重差异化增强 image token 的 logits ==========
    # use_attn = getattr(self, "use_attn", False)
    # use_cfg = getattr(self, "use_cfg", False)
    # alpha = getattr(self, "alpha", 0.0)
    # img_start_idx = getattr(self, "img_start_idx", None)
    # img_end_idx = getattr(self, "img_end_idx", None)
    # importance_weights = getattr(self, "importance_weights", None)
    # important_indices = getattr(self, "important_indices", None)
    # # 注意力增强：根据重要性权重对图像token区域进行差异化增强
    # if use_attn and (img_start_idx is not None) and (img_end_idx is not None) and (importance_weights is not None):
        # 图像token索引需要相对于kv_seq_len进行调整
        # 因为img_start_idx和img_end_idx是基于原始input_ids计算的，但attn_logits使用的是kv_seq_len
        # kv_seq_len = attn_logits.shape[-1]
        # pos_start = img_start_idx
        # pos_end = img_end_idx
        # n_img_tokens_input = pos_end - pos_start - 1
        # input_seq_len = attention_mask.shape[-1] if attention_mask is not None else (bsz and hidden_states.shape[1])
        # # 修复：当 kv_seq_len > input_seq_len 时，说明是在增量生成阶段
        # if kv_seq_len > input_seq_len:
        #     # 在增量生成模式下，图像 tokens 已经存在于 kv cache 中
        #     # 它们的相对位置不会改变，只在整个序列中的绝对位置变了
        #     # 所以 kv_pos_begin 和 kv_pos_end 可以直接用
        #     kv_pos_begin = pos_start + 1
        #     kv_pos_end = pos_end -1
        #     print(f'[ATTN] Incremental generation mode. kv_pos_begin={kv_pos_begin}, kv_pos_end={kv_pos_end}')
        # # 修复：当 kv_seq_len == input_seq_len 时，是第一次前向传播
        # else:
        #     # 原始代码中的比例映射逻辑在增量生成时是错误的，因为它会改变图像 token 的相对位置
        #     kv_pos_begin = pos_start + 1
        #     kv_pos_end = pos_end - 1
        #     print(f'[ATTN] Initial forward pass. kv_pos_begin={kv_pos_begin}, kv_pos_end={kv_pos_end}')
        # # 核心修复: 确保索引范围有效
        # if kv_pos_begin >= kv_pos_end or kv_pos_begin < 0 or kv_pos_end > kv_seq_len:
        #     print(f"[ATTN] Invalid mapped kv range: kv_pos_begin={kv_pos_begin}, kv_pos_end={kv_pos_end}, kv_len={kv_seq_len}. Skipping enhancement.")
        #     pass # 直接跳过增强，使用原始 attn_logits
        # else:
        #     # 修复：在进行切片前，打印信息以验证索引是否正确
        #     print(f'[ATTN] Enhancing logits for tokens from index {kv_pos_begin} to {kv_pos_end}')
        #     # target_logits 的 shape 为 (bsz, num_heads, img_token_count)
        
        # 获取当前查询 token 的索引
        # 在增量生成中，q_len 始终为 1，查询索引是 -1
        # 在初始前向传播中，q_len > 1，查询索引是所有 token
        # 这里我们只对最后一个查询 token 进行增强
    #     target_weight = attn_weights[:, :, -1, img_start_idx+1:img_end_idx-1]
    #     print(f'target_weight:{target_weight.shape}')
    #     # 修复：重要性权重长度处理
    #     num_image_tokens = target_weight.shape[-1]
    #     if len(importance_weights) != num_image_tokens:
    #         print(f'[ATTN] Weight length mismatch: weights={len(importance_weights)}, tokens={num_image_tokens}. Resizing weights.')
    #     # 只对重要性权重大于阈值的token进行适度增强
    #     enhancement_threshold = 1.0
    #     enhancement_mask = importance_weights > enhancement_threshold
    #     print(f'enhancement_mask shape:{enhancement_mask.shape}')
    #     # 确保 importance_weights 的 shape 与 target_logits 匹配
    #     # 广播 importance_weights 到 (1, 1, num_image_tokens)
    #     importance_weights_broadcast = importance_weights.view(1, 1, -1)
    #     if enhancement_mask.any():
    #         # enhanced_logits = target_logits.clone()
    #         # for i in range(target_logits.shape[-1]):
    #         #     importance_factor = min(importance_weights[i].item(), 10)
    #         #     enhancement = target_logits[:, :, i].abs() * alpha + target_logits[:, :, i]
    #         #     enhanced_logits[:, :, i] = enhancement
    #         # # 应用增强后的logits
    #         # attn_logits[:, :, -1, kv_pos_begin:kv_pos_end] = enhanced_logits
    #         enhancement = target_weight.abs() * alpha
    #         enhanced_logits = target_weight + enhancement

    #         # 使用掩码来应用增强
    #         # 扩展掩码以匹配 target_logits 的形状
    #         enhanced_logits = torch.where(
    #             enhancement_mask.view(1, 1, -1),
    #             enhanced_logits,
    #             target_weight
    #         )
    #         # 应用增强后的 logits
    #         attn_weights[:, :, -1, img_start_idx+1:img_end_idx-1] = enhanced_logits
    #     else:
    #         print(f'[ATTN] No tokens meet enhancement threshold {enhancement_threshold}')
    # else:
    #     print(f'[ATTN] No importance-based enhancement applied')
    
    # if torch.isnan(attn_weights).any() or torch.isinf(attn_weights).any():
    #     print("[ATTN] NaN/Inf detected in attn_logits, applying correction...")
    #     # 用有限值替换NaN和Inf
    #     attn_logits = torch.where(
    #         torch.isnan(attn_weights) | torch.isinf(attn_weights),
    #         torch.tensor(torch.finfo(attn_weights.dtype).min, device=attn_weights.device),
    #         attn_weights
    #     )
    
    # # 限制logits的范围以避免数值不稳定
    # attn_logits = torch.clamp(attn_logits, min=-50.0, max=50.0)
    
    
    # 处理softmax后的NaN和Inf
    # if torch.isnan(attn_probs).any() or torch.isinf(attn_probs).any():
    #     print("[ATTN] NaN/Inf detected in attn_probs, applying correction...")
    #     # 用均匀分布替换NaN和Inf
    #     uniform_probs = torch.ones_like(attn_probs) / attn_probs.shape[-1]
    #     attn_probs = torch.where(
    #         torch.isnan(attn_probs) | torch.isinf(attn_probs),
    #         uniform_probs,
    #         attn_weights
    #     )
    # attn output = attn_probs @ V
    # attn_output = torch.matmul(attn_probs, value_states)  # (bsz, num_heads, q_len, head_dim)
    
    
    # upcast attention to fp32
    # attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
    # attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
    # attn_output = torch.matmul(attn_weights, value_states)

    # if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
    #     raise ValueError(
    #         f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
    #         f" {attn_output.size()}"
    #     )
    # # 恢复维度 (bsz, q_len, hidden)
    # attn_output = attn_output.transpose(1, 2).reshape(bsz, q_len, -1)
    # attn_output = self.o_proj(attn_output)

    # # 根据 output_attentions 决定是否返回 attn_probs（注意：返回的 attn_probs dtype 与原实现可能不同）
    # attn_weights = attn_weights if output_attentions else None

    # 根据 use_cache 决定是否要返回缓存（这里直接返回原 past_key_values）
    # return attn_output, attn_weights, past_key_values


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

def qwen_modify_with_importance(model, start_layer: int, end_layer: int, use_attn: bool, alpha: float, use_cfg: bool,
                                img_start_idx: int, img_end_idx: int, importance_weights, important_indices):
    """
    在 model 指定层范围内注入注意力干预参数并替换 forward，使用重要性权重。
    """
    for i in range(start_layer, end_layer):
        attn = model.model.layers[i].self_attn
        attn.use_attn = use_attn
        attn.alpha = alpha
        attn.use_cfg = use_cfg
        attn.img_start_idx = img_start_idx
        attn.img_end_idx = img_end_idx
        attn.layer_idx = i
        # 存储重要性权重信息
        attn.importance_weights = importance_weights.to(model.device)
        attn.important_indices = important_indices.to(model.device)
        attn.forward = types.MethodType(qwen_new_forward_with_importance, attn)
    # 返回模型以便链式调用
    return model
