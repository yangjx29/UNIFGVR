"""
推理干预(inference intervention)
方式：动态替换 Transformers 中的注意力前向，放大图像 token 的注意力

整个调用链是 model.forward() → LlamaModel.forward() → LlamaDecoderLayer.forward() → self_attn.forward()
"""
import math
import types
from typing import Optional, Tuple, Union, Callable, Any
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

# 尝试导入Qwen相关模块
try:
    from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb as qwen_apply_rotary_pos_emb
except ImportError:
    qwen_apply_rotary_pos_emb = None
    warnings.warn("Qwen2 rotary position embedding not available")

try:
    from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
        apply_multimodal_rotary_pos_emb,
        eager_attention_forward 
    )
except ImportError:
    qwen25_apply_mrope = None
    qwen25_eager_attention_forward = None
    warnings.warn("Qwen2.5-VL modules not available")

try:
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
except ImportError:
    QWEN25_ALL_ATTN_FUNCS = {}
    warnings.warn("Attention functions not available")


# def detect_image_tokens(
#     input_ids: Optional[torch.Tensor] = None,
#     position_ids: Optional[torch.Tensor] = None,
#     image_token_mask: Optional[torch.Tensor] = None,
#     img_start_idx: Optional[int] = None,
#     img_end_idx: Optional[int] = None,
#     seq_len: int = 0
# ) -> Tuple[Optional[int], Optional[int]]:
#     """
#     检测图像token的位置范围
    
#     Args:
#         input_ids: 输入token ids
#         position_ids: 位置ids
#         image_token_mask: 图像token掩码 (如果有的话)
#         img_start_idx: 手动指定的图像开始位置
#         img_end_idx: 手动指定的图像结束位置
#         seq_len: 序列长度
    
#     Returns:
#         (start_idx, end_idx): 图像token的起始和结束位置
#     """
#     # 方法1: 使用提供的掩码
#     if image_token_mask is not None:
#         # 找到第一个和最后一个True的位置
#         true_indices = torch.where(image_token_mask)[0]
#         if len(true_indices) > 0:
#             return int(true_indices[0].item()), int(true_indices[-1].item() + 1)
    
#     # 方法2: 使用手动指定的范围
#     if img_start_idx is not None and img_end_idx is not None:
#         if 0 <= img_start_idx < img_end_idx <= seq_len:
#             return img_start_idx, img_end_idx
    
#     # 方法3: 通过特殊token检测 (需要根据具体模型调整)
#     if input_ids is not None:
#         # 这里需要根据具体的图像token ID来检测
#         # 例如: <image> token的ID
#         image_token_ids = [151644, 151645]  # 这些是示例ID，需要根据实际模型调整
#         image_positions = []
#         for token_id in image_token_ids:
#             positions = torch.where(input_ids == token_id)[1]  # 假设input_ids是[batch, seq]
#             image_positions.extend(positions.tolist())
        
#         if image_positions:
#             start_pos = min(image_positions)
#             end_pos = max(image_positions) + 1
#             return start_pos, end_pos
    
#     return None, None


# def apply_attention_enhancement(
#     attn_weights: torch.Tensor,
#     img_start_idx: Optional[int],
#     img_end_idx: Optional[int],
#     alpha: float = 1.0,
#     enhancement_type: str = "multiply"
# ) -> torch.Tensor:
#     """
#     对图像token区域的注意力权重进行增强
    
#     Args:
#         attn_weights: 注意力权重 [batch, heads, q_len, kv_len]
#         img_start_idx: 图像token开始位置
#         img_end_idx: 图像token结束位置
#         alpha: 增强系数
#         enhancement_type: 增强类型 ("multiply", "add", "replace")
    
#     Returns:
#         增强后的注意力权重
#     """
#     if img_start_idx is None or img_end_idx is None:
#         return attn_weights
    
#     # 确保索引在有效范围内
#     kv_len = attn_weights.size(-1)
#     img_start_idx = max(0, min(img_start_idx, kv_len))
#     img_end_idx = max(img_start_idx, min(img_end_idx, kv_len))
    
#     if img_start_idx >= img_end_idx:
#         return attn_weights
    
#     # 创建增强后的注意力权重
#     enhanced_weights = attn_weights.clone()
#     image_region = enhanced_weights[:, :, :, img_start_idx:img_end_idx]
    
#     if enhancement_type == "multiply":
#         # 乘法增强
#         enhanced_weights[:, :, -1, img_start_idx:img_end_idx] = image_region * alpha
#     elif enhancement_type == "add":
#         # 加法增强
#         enhanced_weights[:, :, -1, img_start_idx:img_end_idx] = image_region + (image_region.abs() * alpha)
#     elif enhancement_type == "replace":
#         # 替换增强
#         enhanced_weights[:, :, -1, img_start_idx:img_end_idx] = image_region.abs() * alpha
#     else:
#         warnings.warn(f"Unknown enhancement type: {enhancement_type}, using multiply")
#         enhanced_weights[:, :, :, img_start_idx:img_end_idx] = image_region * alpha
    
#     return enhanced_weights


"""
这里是qwen的源码
def forward(
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
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_multimodal_rotary_pos_emb(
            query_states, key_states, cos, sin, self.rope_scaling["mrope_section"]
        )

        if past_key_values is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}  # Specific to RoPE models
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_output, attn_weights = attention_interface(
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

        attn_output = attn_output.reshape(bsz, q_len, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights
"""


def qwen_modify_attention(
    model: Any,
    start_layer: int = 0,
    end_layer: int = -1,
    use_attn: bool = True,
    alpha: float = 1.5,
    use_cfg: bool = False,
    img_start_idx: Optional[int] = None,
    img_end_idx: Optional[int] = None,
    enhancement_type: str = "multiply",
    image_token_ids: Optional[list] = None,
    verbose: bool = True
) -> bool:
    """
    为 Qwen2 系模型注入注意力放大开关与参数（推理期）。
    
    Args:
        model: 要修改的模型
        start_layer: 开始修改的层索引
        end_layer: 结束修改的层索引 (-1表示到最后一层)
        use_attn: 是否启用注意力增强
        alpha: 注意力增强系数
        use_cfg: 是否在CFG模式下使用
        img_start_idx: 图像token开始位置
        img_end_idx: 图像token结束位置
        enhancement_type: 增强类型 ("multiply", "add", "replace")
        image_token_ids: 图像token的ID列表，用于自动检测
        verbose: 是否打印详细信息
    
    Returns:
        bool: 是否成功修改
    """
    if not use_attn:
        if verbose:
            print("wrong 注意力增强已禁用")
        return True
    
    # 检查模型结构
    if not hasattr(model, 'model') or not hasattr(model.model, 'language_model'):
        if verbose:
            print("wrong 模型结构不支持，跳过修改")
        return False
    
    # 确定层范围
    total_layers = len(model.model.language_model.layers)
    if end_layer == -1:
        end_layer = total_layers
    end_layer = min(end_layer, total_layers)
    
    if start_layer >= end_layer:
        if verbose:
            print(f"wrong 无效的层范围: {start_layer}-{end_layer}")
        return False
    
    if verbose:
        print(f"修改层范围: {start_layer}-{end_layer}, 增强系数: {alpha}")
    
    # 尝试导入Qwen2Attention
    try:
        from transformers.models.qwen2.modeling_qwen2 import Qwen2Attention
    except ImportError:
        if verbose:
            print("wrong transformers 未包含 Qwen2Attention，尝试其他方式")
        # 尝试其他可能的导入路径
        try:
            from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2VLAttention as Qwen2Attention
        except ImportError:
            if verbose:
                print("wrong 无法找到Qwen注意力模块，跳过注入")
            return False

    # def qwen_new_forward(self,
    #                      hidden_states: torch.Tensor,
    #                      attention_mask: Optional[torch.Tensor] = None,
    #                      position_ids: Optional[torch.Tensor] = None,
    #                      past_key_values: Optional[Any] = None,
    #                      output_attentions: bool = False,
    #                      use_cache: bool = False,
    #                      cache_position: Optional[torch.Tensor] = None,
    #                      position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    #                      **kwargs) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    #     """
    #     增强的注意力前向传播函数，支持图像token注意力增强
    #     """
    #     bsz, q_len, _ = hidden_states.size()
        
    #     # 获取配置参数
    #     use_attn_flag = getattr(self, 'use_attn', False)
    #     use_cfg_flag = getattr(self, 'use_cfg', False)
    #     alpha_v = getattr(self, 'alpha', 1.0)
    #     enhancement_type = getattr(self, 'enhancement_type', 'multiply')
    #     image_token_ids = getattr(self, 'image_token_ids', None)
    #     verbose = getattr(self, 'verbose', False)
        
    #     if verbose:
    #         print(f'[ATTN] Layer {getattr(self, "layer_idx", "unknown")}: bsz={bsz}, q_len={q_len}')
        
    #     # Step 1: 线性变换得到 QKV
    #     query_states = self.q_proj(hidden_states)
    #     key_states = self.k_proj(hidden_states)
    #     value_states = self.v_proj(hidden_states)

    #     # 重塑为 multi-head 形式
    #     query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    #     key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    #     value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

    #     # Step 2: 应用 RoPE
    #     if position_embeddings is None:
    #         # 尝试从kwargs获取
    #         position_embeddings = kwargs.get('position_embeddings', None)
        
    #     if position_embeddings is None:
    #         if verbose:
    #             print("[ATTN] 警告: 缺少 position_embeddings，跳过 RoPE")
    #     else:
    #         cos, sin = position_embeddings
    #         # 应用 RoPE
    #         if qwen25_apply_mrope is not None:
    #             try:
    #                 query_states, key_states = qwen25_apply_mrope(
    #                     query_states, key_states, cos, sin, 
    #                     self.rope_scaling.get("mrope_section", None), 
    #                     unsqueeze_dim=1
    #                 )
    #             except Exception as e:
    #                 if verbose:
    #                     print(f"wrong RoPE 应用失败: {e}")
    #         elif qwen_apply_rotary_pos_emb is not None:
    #             try:
    #                 query_states, key_states = qwen_apply_rotary_pos_emb(
    #                     query_states, key_states, cos, sin
    #                 )
    #             except Exception as e:
    #                 if verbose:
    #                     print(f"wrong 传统 RoPE 应用失败: {e}")

    #     # Step 3: 更新 KV Cache
    #     if past_key_values is not None:
    #         try:
    #             cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
    #             key_states, value_states = past_key_values.update(
    #                 key_states, value_states, getattr(self, 'layer_idx', 0), cache_kwargs
    #             )
    #         except Exception as e:
    #             if verbose:
    #                 print(f"wrong KV Cache 更新失败: {e}")

    #     # Step 4: 调用注意力核心函数（支持 eager / flash_attn）
    #     attention_interface = qwen25_eager_attention_forward
    #     if qwen25_eager_attention_forward is None:
    #         # 回退到标准注意力
    #         from transformers.modeling_utils import eager_attention_forward
    #         attention_interface = eager_attention_forward
        
    #     if getattr(self.config, '_attn_implementation', 'eager') != 'eager':
    #         if self.config._attn_implementation in QWEN25_ALL_ATTN_FUNCS:
    #             attention_interface = QWEN25_ALL_ATTN_FUNCS[self.config._attn_implementation]
        
    #     try:
    #         attn_output, attn_weights = attention_interface(
    #             self,
    #             query_states,
    #             key_states,
    #             value_states,
    #             attention_mask,
    #             dropout=0.0 if not self.training else getattr(self, 'attention_dropout', 0.0),
    #             scaling=getattr(self, 'scaling', 1.0),
    #             sliding_window=getattr(self, 'sliding_window', None),
    #             position_ids=position_ids,
    #             **kwargs,
    #         )
    #     except Exception as e:
    #         if verbose:
    #             print(f"wrong 注意力计算失败: {e}")
    #         # 回退到简单实现
    #         attn_weights = torch.matmul(query_states, key_states.transpose(-2, -1)) / math.sqrt(self.head_dim)
    #         if attention_mask is not None:
    #             attn_weights = attn_weights + attention_mask
    #         attn_weights = F.softmax(attn_weights, dim=-1)
    #         attn_output = torch.matmul(attn_weights, value_states)

    #     # Step 5: 图像token注意力增强
    #     if (use_attn_flag and not use_cfg_flag and 
    #         attn_weights is not None and q_len > 0):
            
    #         # 检测图像token位置
    #         img_s, img_e = detect_image_tokens(
    #             input_ids=kwargs.get('input_ids'),
    #             position_ids=position_ids,
    #             image_token_mask=kwargs.get('image_token_mask'),
    #             img_start_idx=getattr(self, 'img_start_idx', None),
    #             img_end_idx=getattr(self, 'img_end_idx', None),
    #             seq_len=key_states.size(2)
    #         )
            
    #         if img_s is not None and img_e is not None:
    #             if verbose:
    #                 print(f'[ATTN] 增强图像token注意力: {img_s}-{img_e}, alpha={alpha_v}')
                
    #             # 应用注意力增强
    #             attn_weights = apply_attention_enhancement(
    #                 attn_weights, img_s, img_e, alpha_v, enhancement_type
    #             )
                
    #             # 重新计算注意力输出
    #             attn_output = torch.matmul(attn_weights, value_states)
    #         else:
    #             print(f'img_s is None or img_e is None, imgs:{img_s}, imge:{img_e}')
    #     # Step 6: 输出投影
    #     attn_output = attn_output.reshape(bsz, q_len, -1).contiguous()
    #     attn_output = self.o_proj(attn_output)
        
    #     if not output_attentions:
    #         attn_weights = None
            
    #     return attn_output, attn_weights
    def qwen_new_forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        # Step 1: QKV 投影
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # Reshape to multi-head format
        query_states = query_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)

        # Step 2: 多模态 RoPE (MRoPE)
        if position_embeddings is None:
            raise ValueError("position_embeddings must be provided for Qwen2.5-VL.")
        cos, sin = position_embeddings
        rope_section = self.rope_scaling.get("mrope_section", None)
        query_states, key_states = apply_multimodal_rotary_pos_emb(
            query_states, key_states, cos, sin, position_ids, rope_section
        )

        # Step 3: KV Cache 更新
        if past_key_values is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)

        # Step 4: 使用 eager 或 Flash Attention 实现计算 attn_output 和 attn_weights
        # TODO这里不能直接用eager，因为eager是已经 softmax 过了
        attention_interface = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementatio·n]

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            sliding_window=self.sliding_window,
            position_ids=position_ids,
            **kwargs,
        )
        # 注意：attn_weights 是 softmax 前的 logits！这是关键！

        # --- 🔥 PAI's modification: 放大最后一个 token 对 image tokens 的注意力 ---
        # 读取注入的控制参数（来自 llama_modify）
        use_attn = getattr(self, "use_attn", False)
        use_cfg = getattr(self, "use_cfg", False)
        img_start_idx = getattr(self, "img_start_idx", None)
        img_end_idx = getattr(self, "img_end_idx", None)
        alpha = getattr(self, "alpha", 0.0)

        if use_attn and (not use_cfg) and attn_weights is not None and q_len > 0:
            if img_start_idx is not None and img_end_idx is not None:
                kv_seq_len = attn_weights.size(-1)
                if img_start_idx < kv_seq_len and img_end_idx <= kv_seq_len and img_start_idx < img_end_idx:
                    # 只修改最后一个 query token 的 scores
                    device = attn_weights.device
                    dtype = attn_weights.dtype
                    slice_ = attn_weights[:, :, -1, img_start_idx:img_end_idx]  # [B, H, LEN_IMG]

                    # 核心公式：|w| * alpha + w
                    boosted_slice = slice_.abs() * alpha + slice_
                    attn_weights[:, :, -1, img_start_idx:img_end_idx] = boosted_slice.to(dtype)

        # --- 🔥 END OF PAI's modification ---

        # Step 5: Softmax 归一化（由 attention_interface 内部完成？视实现而定）
        # ⚠️ 注意：eager_attention_forward 返回的是 softmax 后的结果吗？
        # 根据 HF 实现，通常返回的是 softmax 后的概率分布。
        # 所以我们上面的操作是在 softmax 前做的吗？❌ 不一定！

        # ❗因此我们必须确认：attn_weights 是 logits 还是 probs
        # 如果是 probs，则不能这样改！必须 hook 到更底层！

        # 🛠️ 安全起见：假设它是 logits（如某些版本），否则此修改无效。
        # 更推荐做法：使用自定义 attention_interface，但太复杂。

        # 当前策略：信任 attn_weights 是 softmax 前的 logits（部分 HF 实现如此）

        # Step 6: reshape 输出
        attn_output = attn_output.reshape(bsz, q_len, -1).contiguous()
        attn_output = self.o_proj(attn_output)

        return attn_output, attn_weights

    # 修改指定层的参数
    success_count = 0
    for i in range(start_layer, end_layer):
        try:
            attn = model.model.language_model.layers[i].self_attn
            if verbose:
                print(f'修改第 {i} 层注意力模块\natten:{attn}')
            
            # 设置增强参数
            attn.use_attn = use_attn
            attn.alpha = alpha
            attn.use_cfg = use_cfg
            attn.img_start_idx = img_start_idx
            attn.img_end_idx = img_end_idx
            attn.enhancement_type = enhancement_type
            attn.image_token_ids = image_token_ids
            attn.verbose = verbose
            attn.layer_idx = i
            
             # 动态替换模型中第i层self-attention的forward方法。 绑定后，调用时 self 会自动指向 model.model.layers[i].self_attn
            attn.forward = types.MethodType(qwen_new_forward, attn)
            success_count += 1
            
        except Exception as e:
            if verbose:
                print(f'wrong第 {i} 层修改失败: {e}')
            continue
    
    if verbose:
        print(f'[ATTN] 成功修改 {success_count}/{end_layer - start_layer} 层')
    
    return success_count > 0


