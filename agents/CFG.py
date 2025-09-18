"""
输出精修（logits refine）
以 logits_processor 的形式在 model.generate() 中注入，做“多模态 - 纯文本”的logits差分。建议配合 nucleus sampling。
现在需要修改这个文件适配Qwen
"""
# 上述文档字符串：说明本文件实现的 CFG（Classifier-Free Guidance）式 logits 精修逻辑
import torch  # 张量运算与数值操作
import torch.nn.functional as F  # 常用函数（log_softmax 等）
from transformers import (
    LogitsProcessor,  # HF 生成流程中的 logits 处理器基类
)


class CFGLogits(LogitsProcessor):  # 继承自 LogitsProcessor，用于生成时动态调整 logits
    """
    gamma：差分强度（如 1.1，增大可更强抑制文本惯性）。
    neg_promt：构造“无图场景”的输入（去掉图像 token 的文本提示）。
    llm_model：对应的纯文本模型或同一模型在“无图”路径上的引用。
     Qwen 版本不再依赖内部层标志位，按步重用 past_key_values 实现高效无图分支
    """
    def __init__(
        self,
        guidance_scale,  # 指导强度（对应 README 中 gamma 概念）
        uncond_inputs,  # 无条件输入的完整字典（含 input_ids / attention_mask）
        model,  # 被调用的模型（与主模型共享缓存）
        image=None,  # 预留：如需图像路径/张量
        input_type="inputs_ids",  # 指定无条件路径的输入类型
    ):
        self.guidance_scale = guidance_scale  # 保存指导强度
        self.uncond_inputs = uncond_inputs  # 保存无条件输入字典
        self.model = model  # 保存模型引用
        self.image = image  # 备用字段
        self.out = None  # 缓存无条件分支的输出（含 past_key_values）
        self.input_type = input_type  # 输入类型：inputs_ids 或 inputs_embeds

    def __call__(self, input_ids, scores):  # 生成每步都会调用：接收当前 step 的 input_ids 与原始 logits
        scores = F.log_softmax(scores, dim=-1)  # 将当前 logits 变换到对数概率域
        if self.guidance_scale == 1:  # 指导强度为 1：等价不做 CFG
            return scores
        # TODO 迁移PAI: Qwen 路线——不改内部层，直接跑无图分支并复用缓存
        if self.out is None:  # 第一次：完整无图prompt前向，建立past_key_values
            if self.input_type == "inputs_ids":
                self.out = self.model(
                    input_ids=self.uncond_inputs.get('input_ids'),
                    attention_mask=self.uncond_inputs.get('attention_mask'),
                    use_cache=True,
                )
            elif self.input_type == "inputs_embeds":
                self.out = self.model(
                    inputs_embeds=self.uncond_inputs.get('inputs_embeds'),
                    attention_mask=self.uncond_inputs.get('attention_mask'),
                    use_cache=True,
                )
            else:
                print("[PAI][CFG] Neither input_ids nor inputs_embeds is provided.")
        else:  # 后续步：仅追加最后一个token，复用无图路径缓存
            self.out = self.model(
                input_ids=input_ids[:, -1:],
                use_cache=True,
                past_key_values=self.out.past_key_values,
            )

        unconditional_logits = F.log_softmax(self.out.logits[:, -1, :], dim=-1)  # 无条件分支的对数概率

        cutoff = torch.log(torch.tensor(0.1, device=scores.device, dtype=scores.dtype)) + scores.max(dim=-1, keepdim=True).values  # 截断阈值（避免尾部噪声）
        out = (
            self.guidance_scale * (scores - unconditional_logits) + unconditional_logits
        )  # CFG 公式：扩大条件-无条件差分，再加回无条件
        cd_logits = out.masked_fill(scores < cutoff, -float("inf"))  # 对低于阈值的项置为 -inf
        return cd_logits  # 返回处理后的 logits（对数域），供采样/搜索使用
