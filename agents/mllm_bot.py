import sys
import os

# 确保导入正确的utils模块
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# 直接导入UniFGVR目录下的utils模块
sys.path.insert(0, '/data/yjx/MLLM/UniFGVR')
from utils.util import encode_base64, prepare_qwen2_5_input, get_important_image_tokens, create_attention_mask

import torch
from os import path
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
from PIL import Image
from agents.CFG import CFGLogits 
from agents.attention import qwen_modify, qwen_modify_with_importance
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from skimage.measure import block_reduce

QWEN = {
    'Qwen2.5-VL-7B': 'Qwen/Qwen2.5-VL-7B-Instruct'
}

ANSWER_INSTRUCTION = 'Answer given questions. If you are not sure about the answer, say you don\'t ' \
                     'know honestly. Don\'t imagine any contents that are not in the image.'

SUB_ANSWER_INSTRUCTION = 'Answer: '  # template following qwen2_5 huggingface demo


def get_chat_log(questions, answers, last_n=-1):
    n_addition_q = len(questions) - len(answers)
    assert (n_addition_q) in [0, 1]
    template = 'Question: {} \nAnswer: {} \n'
    chat_log = ''
    if last_n > 0:
        answers = answers[-last_n:]
        questions = questions[-(last_n + n_addition_q):]
    elif last_n == 0:
        answers = []
        questions = questions[-1:] if n_addition_q else []

    for i in range(len(answers)):
        chat_log = chat_log + template.format(questions[i], answers[i])
    if n_addition_q:
        chat_log = chat_log + 'Question: {}'.format(questions[-1])
    else:
        chat_log = chat_log[:-2]
    return chat_log


def trim_answer(answer):
    if isinstance(answer, list):
        return answer
    answer = answer.split('Question:')[0].replace('\n', ' ').strip()
    return answer


class MLLMBot:
    def __init__(self, model_tag, model_name, pai_enable_attn=False, device='cpu', device_id=0, bit8=False, max_answer_tokens=-1):
        self.model_tag = model_tag
        self.model_name = model_name
        self.max_answer_tokens = max_answer_tokens
        local_model_path_abs = "/home/Dataset/Models/Qwen"
        local_model_path = path.join(local_model_path_abs, QWEN[self.model_tag].split('/')[-1])
        self.qwen2_5_processor = AutoProcessor.from_pretrained(local_model_path)
        # self.qwen2_5_processor = AutoProcessor.from_pretrained(
        #     local_model_path,
        #     trust_remote_code=True,
        #     padding_side='left',
        #     use_fast=True,
        # )
        if device == 'cpu':
            self.device = 'cpu'
            self.qwen2_5 = Qwen2_5_VLForConditionalGeneration.from_pretrained(local_model_path)
        else:
            self.device = 'cuda:{}'.format(device_id)
            self.bit8 = bit8
            dtype = {'load_in_8bit': True} if self.bit8 else {'torch_dtype': torch.float16}
            # attn_implementation="sdpa"与 output_attentions不兼容
            # self.qwen2_5 = Qwen2_5_VLForConditionalGeneration.from_pretrained(local_model_path,
            #                                                         device_map={'': int(device_id)},
            #                                                         **dtype)
            self.qwen2_5 = Qwen2_5_VLForConditionalGeneration.from_pretrained(local_model_path,
                                                                    device_map="auto",
                                                                    # attn_implementation="eager",
                                                                    torch_dtype=torch.float32,
                                                                    ).eval()
        # print(f'model:{self.qwen2_5}')
        print(f'local_model_path: {local_model_path}')
        
        # TODO超参数
        self.pai_enable_attn = pai_enable_attn   # 阶段一：是否增强图像注意力
        self.pai_alpha = 0.5           # 阶段一：增强系数 α
        self.pai_layers = (10, 28)     # 阶段一：层先验（深层更有效）
        self.pai_enable_cfg = False    # 阶段二：是否开启CFG logits精炼
        self.pai_gamma = 1.1           # 阶段二：γ 指导强度
        self.num_map = 0
        
    # # TODO 这里应该需要考虑chunk切分
    # def _resolve_img_token_span(self, messages, inputs):
    #     """返回(img_start_idx, img_end_idx)。
    #     启发式：缺少显式 image special token 时，近似把末尾 256 个 token 当作图像区域。
    #     若序列过短或无法解析，则返回 (None, None) 跳过注入。
    #     """
    #     try:
    #         input_ids = inputs.input_ids
    #         if input_ids is None:
    #             print(f'input_ids is None')
    #             return None, None
    #         seq_len = input_ids.shape[1]
    #         img_tokens = 256
    #         print(f'input_ids:{input_ids.shape}\nseq_len:{seq_len}')
    #         if seq_len <= img_tokens:
    #             print(f'seq_len <= img_tokens')
    #             return None, None
    #         img_start = seq_len - img_tokens
    #         img_end = seq_len
    #         print(f'img_start:{img_start}, img_end:{img_end}')
    #         return img_start, img_end
    #     except Exception as e:
    #         print(f"error return None None:{e}")
    #         return None, None


    def _resolve_img_token_span(self, messages, inputs):
        try:
            input_ids = inputs.input_ids
            if input_ids is None:
                print(f'input_ids is None')
                return None, None
            seq_len = input_ids.shape[1]
            # tokenizer 里有 special token 的映射
            tokenizer = self.qwen2_5_processor.tokenizer
            # img_start_token_id = tokenizer.convert_tokens_to_ids("<img_start>")
            # img_end_token_id   = tokenizer.convert_tokens_to_ids("<img_end>")
            vision_start_id = tokenizer.convert_tokens_to_ids('<|vision_start|>')
            image_pad_id = tokenizer.convert_tokens_to_ids('<|image_pad|>')
            vision_end_id = tokenizer.convert_tokens_to_ids('<|vision_end|>')
            print(f'input_ids:{input_ids.shape}\nseq_len:{seq_len}')
            input_ids_list = input_ids[0].tolist()
            if vision_start_id in input_ids_list and vision_end_id in input_ids_list:
                img_start = input_ids_list.index(vision_start_id)
                img_end   = input_ids_list.index(vision_end_id) + 1  # 包含 img_end
                print(f"找到 image token span: img_start={img_start}, img_end={img_end}")
                return img_start, img_end
            else:
                print("未找到 image token span")
                return None, None
        except Exception as e:
            print(f"error return None None:{e}")
            return None, None

    def _inject_qwen_pai_attention(self, img_start_idx, img_end_idx):
        if img_start_idx is None or img_end_idx is None:
            print('[ATTN] skip injection for Qwen (img span unresolved).')
            return
        print(f'[ATTN] inject Qwen attention layers {self.pai_layers} alpha={self.pai_alpha} span=({img_start_idx},{img_end_idx})')
        qwen_modify(self.qwen2_5, self.pai_layers[0], self.pai_layers[1], True, self.pai_alpha, False, img_start_idx, img_end_idx)

    def _inject_qwen_pai_attention_with_importance(self, img_start_idx, img_end_idx, important_tokens_info):
        if img_start_idx is None or img_end_idx is None:
            print('[ATTN] skip injection for Qwen (img span unresolved).')
            return
        
        print(f'[ATTN] inject Qwen attention layers with importance weights {self.pai_layers} alpha={self.pai_alpha} span=({img_start_idx},{img_end_idx})')
        
        # 提取重要性权重信息
        importance_weights = important_tokens_info['weights']  # 所有图像token的权重
        important_indices = important_tokens_info['important_indices']  # 重要token的索引
        
        # 调用修改函数，传递重要性信息
        qwen_modify_with_importance(self.qwen2_5, self.pai_layers[0], self.pai_layers[1], True, self.pai_alpha, False, img_start_idx, img_end_idx, importance_weights, important_indices)

    def get_name(self):
        return self.model_name

    def __call_qwen2_5(self, raw_image, prompt):
        # print(f"MLLMBot prompt: {prompt}")

        if isinstance(raw_image, Image.Image):
            raw_image = [raw_image]

        content = []
        # 先添加所有图像
        # for img in raw_image:
        #     content.append({"type": "image", "image": img})
        # content.append({"type": "text", "text": prompt})
        for img in raw_image:
            image_str = encode_base64(img)
            content.append({"type": "image", "image": f'data:image;base64,{image_str}'})
        content.append({"type": "text", "text": prompt})
        # 1. 构造 messages
        messages = [
            {
                "role": "user",
                "content": content
            }
        ]
        # # 2. 构造输入文本
        # text = self.qwen2_5_processor.apply_chat_template(
        #     messages, tokenize=False, add_generation_prompt=True
        # )
        # # 3. 提取视觉输入
        # image_inputs, video_inputs = process_vision_info(messages)

        # if self.device == 'cpu':
        #     inputs = self.qwen2_5_processor(
        #         text=[text],    
        #         images=image_inputs,  
        #         videos=video_inputs,           
        #         padding=True,
        #         return_tensors="pt"
        #     )
        # else:
        #     inputs = self.qwen2_5_processor(
        #         text=[text],    
        #         images=image_inputs,  
        #         videos=video_inputs,           
        #         padding=True,
        #         return_tensors="pt"
        #     ).to(self.device,torch.float16)
        # generated_ids = self.qwen2_5.generate(**inputs, max_new_tokens=128)
        inputs = prepare_qwen2_5_input(messages, self.qwen2_5_processor).to(self.device, torch.bfloat16)

        # TODO阶段一 注意力增强
        if self.pai_enable_attn:
            # 1.计算rel-attention计算注意力矩阵
            from agents.qwen2_5_methods import rel_attention_qwen2_5
            general_question = 'Write a general description of the image.'
            # 确保两个prompt格式一致
            formatted_prompt = f"<image>\nUSER: {prompt} Answer the question with a concise phrase.\nASSISTANT:"
            general_prompt = f"<image>\nUSER: {general_question} Answer the question with a concise phrase.\nASSISTANT:"
            att_map = rel_attention_qwen2_5(raw_image[0], formatted_prompt, general_prompt, self.qwen2_5, self.qwen2_5_processor)
            import matplotlib.pyplot as plt
            path='/data/yjx/MLLM/UniFGVR/maps/'
            os.makedirs(path, exist_ok=True)
            plt.imshow(att_map, interpolation='none')
            plt.axis('off')
            plt.savefig(
                fname=f"{path}/attention_map_{self.num_map}.png",  # 保存为 PNG 格式（推荐，无损）
                dpi=300,                    
                bbox_inches='tight',        # 自动去除多余空白
                pad_inches=0                # 无额外边距
            )
            self.num_map +=1

            # 2.将att_map中的重要token和原始输入token位置对应
            # 将注意力权重映射到原始输入中的图像token

            important_tokens_info = get_important_image_tokens(att_map, inputs, self.qwen2_5_processor, threshold=1)
            print(f"图像token位置范围: {important_tokens_info['positions']}")
            print(f"重要stoken数量: {len(important_tokens_info['important_indices'])}")
            if len(important_tokens_info['important_indices']) > 0:
                print(f"最高注意力权重: {important_tokens_info['important_weights'].max().item():.4f}")
                print(f"平均注意力权重: {important_tokens_info['important_weights'].mean().item():.4f}")
            # 创建注意力mask
            # attention_mask = create_attention_mask(att_map, inputs, important_tokens_info, self.qwen2_5_processor, threshold=1)
            img_start_idx, img_end_idx = self._resolve_img_token_span(messages, inputs)
            
            # 将注意力重要性信息传递给注意力层
            # self._inject_qwen_pai_attention(img_start_idx, img_end_idx)
            self._inject_qwen_pai_attention_with_importance(img_start_idx, img_end_idx, important_tokens_info)

        # TODO阶段二 CFG logits 精炼（在generate中注入LogitsProcessor）
        logits_processors = None
        if self.pai_enable_cfg:
            try:
                # 构造无图输入：移除images，仅保留文本
                uncond_messages = [
                    {"role": "user", "content": [{"type": "text", "text": prompt}]}
                ]
                uncond_text = self.qwen2_5_processor.apply_chat_template(
                    uncond_messages, tokenize=False, add_generation_prompt=True
                )
                if self.device == 'cpu':
                    uncond_inputs = self.qwen2_5_processor(
                        text=[uncond_text], images=None, videos=None, padding=True, return_tensors="pt"
                    )
                else:
                    uncond_inputs = self.qwen2_5_processor(
                        text=[uncond_text], images=None, videos=None, padding=True, return_tensors="pt"
                    ).to(self.device, torch.float16)

                logits_processors = [
                    CFGLogits(
                        guidance_scale=self.pai_gamma,
                        uncond_inputs={
                            'input_ids': uncond_inputs.input_ids,
                            'attention_mask': getattr(uncond_inputs, 'attention_mask', None)
                        },
                        model=self.qwen2_5,
                        image=None,
                        input_type="inputs_ids",
                    )
                ]
            except Exception as e:
                print(f'[PAI][CFG] fallback without CFG due to: {e}')
                logits_processors = None

        with torch.no_grad():
            generated_ids = self.qwen2_5.generate(
                **inputs,
                max_new_tokens=256,
                # logits_processor=logits_processors
            )
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]

        # 8. 解码
        reply = self.qwen2_5_processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )
        # print(f"test MLLM answer after decode: {reply}")
        return reply

    def answer_chat_log(self, raw_image, chat_log, n_qwen2_5_context=-1):
        # prepare the context for qwen2_5
        qwen2_5_prompt = '\n'.join([ANSWER_INSTRUCTION,
                                  get_chat_log(chat_log['questions'],chat_log['answers'],
                                               last_n=n_qwen2_5_context), SUB_ANSWER_INSTRUCTION]
                                 )

        reply = self.__call_qwen2_5(raw_image, qwen2_5_prompt)
        trimmed_reply = trim_answer(reply)
        return reply, trimmed_reply

    def tell_me_the_obj(self, raw_image, super_class, super_unit):
        std_prompt = f"Questions: What is the {super_unit} of the {super_class} in this photo? Answer:"
        # std_prompt = f"Questions: What is the name of the main object in this photo? Answer:"
        reply = self.__call_qwen2_5(raw_image, std_prompt)
        trimmed_reply = trim_answer(reply)
        return reply, trimmed_reply

    def describe_attribute(self, raw_image, attr_prompt):
        # raw_image是Image.open之后的格式
        reply = self.__call_qwen2_5(raw_image, attr_prompt)
        trimmed_reply = trim_answer(reply)
        return reply, trimmed_reply
    
    def compare_attention_enhancement(self, raw_image, attr_prompt, save_dir="/data/yjx/MLLM/UniFGVR/experiments/attention_comparison"):
        """
        对比注意力增强前后的效果
        """
        import os
        os.makedirs(save_dir, exist_ok=True)
        
        print("=" * 60)
        print("ATTENTION ENHANCEMENT COMPARISON")
        print("=" * 60)
        
        # 1. 运行未增强版本
        print("\n[1] Running WITHOUT attention enhancement...")
        original_attn = self.pai_enable_attn
        self.pai_enable_attn = False
        
        reply_no_enhance, _ = self.describe_attribute(raw_image, attr_prompt)
        print(f"Without enhancement: {reply_no_enhance}")
        
        # 2. 运行增强版本
        print("\n[2] Running WITH attention enhancement...")
        self.pai_enable_attn = True
        
        reply_with_enhance, _ = self.describe_attribute(raw_image, attr_prompt)
        print(f"With enhancement: {reply_with_enhance}")
        
        # 3. 恢复原始设置
        self.pai_enable_attn = original_attn
        
        # 4. 保存对比结果
        with open(os.path.join(save_dir, "comparison_results.txt"), "w", encoding="utf-8") as f:
            f.write("ATTENTION ENHANCEMENT COMPARISON\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Prompt: {attr_prompt}\n\n")
            f.write(f"Without enhancement: {reply_no_enhance}\n\n")
            f.write(f"With enhancement: {reply_with_enhance}\n\n")
            f.write(f"Enhancement layers: {self.pai_layers}\n")
            f.write(f"Alpha value: {self.pai_alpha}\n")
        
        print(f"\n[3] Comparison results saved to {save_dir}")
        print("=" * 60)
        
        return reply_no_enhance, reply_with_enhance

    def caption(self, raw_image):
        # starndard way to caption an image in the qwen2_5 paper
        std_prompt = 'a photo of'
        reply = self.__call_qwen2_5(raw_image, std_prompt)
        reply = reply.replace('\n', ' ').strip()  # trim caption
        return reply

    def call_llm(self, prompts):
        prompts_temp = self.qwen2_5_processor(None, prompts, return_tensors="pt")
        input_ids = prompts_temp['input_ids'].to(self.device)
        attention_mask = prompts_temp['attention_mask'].to(self.device, torch.float16)

        prompts_embeds = self.qwen2_5.language_model.get_input_embeddings()(input_ids)

        with torch.no_grad():
            outputs = self.qwen2_5.language_model.generate(
                inputs_embeds=prompts_embeds,
                attention_mask=attention_mask)

        outputs = self.qwen2_5_processor.decode(outputs[0], skip_special_tokens=True)
        return outputs
