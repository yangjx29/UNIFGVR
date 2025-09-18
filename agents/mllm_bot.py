import torch
from os import path
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
from PIL import Image
from agents.CFG import CFGLogits 
from agents.attention import qwen_modify  

QWEN = {
    'Qwen2.5-VL-8B': 'Qwen/Qwen2.5-VL-7B-Instruct'
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
    def __init__(self, model_tag, model_name, device='cpu', device_id=0, bit8=False, max_answer_tokens=-1):
        self.model_tag = model_tag
        self.model_name = model_name
        self.max_answer_tokens = max_answer_tokens
        local_model_path_abs = "/home/Dataset/Models/Qwen"
        local_model_path = path.join(local_model_path_abs, QWEN[self.model_tag].split('/')[-1])
        self.qwen2_5_processor = AutoProcessor.from_pretrained(local_model_path)
        if device == 'cpu':
            self.device = 'cpu'
            self.qwen2_5 = Qwen2_5_VLForConditionalGeneration.from_pretrained(local_model_path)
        else:
            self.device = 'cuda:{}'.format(device_id)
            self.bit8 = bit8
            dtype = {'load_in_8bit': True} if self.bit8 else {'torch_dtype': torch.float16}
            self.qwen2_5 = Qwen2_5_VLForConditionalGeneration.from_pretrained(local_model_path,
                                                                    device_map={'': int(device_id)},
                                                                    **dtype)
            
        print(f'local_model_path: {local_model_path}')
        
        # TODO超参数
        self.pai_enable_attn = True   # 阶段一：是否增强图像注意力
        self.pai_alpha = 0.2           # 阶段一：增强系数 α
        self.pai_layers = (14, 28)     # 阶段一：层先验（深层更有效）
        self.pai_enable_cfg = False    # 阶段二：是否开启CFG logits精炼
        self.pai_gamma = 1.1           # 阶段二：γ 指导强度
        
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

    def get_name(self):
        return self.model_name

    def __call_qwen2_5(self, raw_image, prompt):
        # print(f"MLLMBot prompt: {prompt}")

        if isinstance(raw_image, Image.Image):
            raw_image = [raw_image]

        content = []
        # 先添加所有图像
        for img in raw_image:
            content.append({"type": "image", "image": img})
        content.append({"type": "text", "text": prompt})
        # 1. 构造 messages
        messages = [
            {
                "role": "user",
                "content": content
            }
        ]
        # 2. 构造输入文本
        text = self.qwen2_5_processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        # 3. 提取视觉输入
        image_inputs, video_inputs = process_vision_info(messages)

        if self.device == 'cpu':
            inputs = self.qwen2_5_processor(
                text=[text],    
                images=image_inputs,  
                videos=video_inputs,           
                padding=True,
                return_tensors="pt"
            )
        else:
            inputs = self.qwen2_5_processor(
                text=[text],    
                images=image_inputs,  
                videos=video_inputs,           
                padding=True,
                return_tensors="pt"
            ).to(self.device,torch.float16)
        # generated_ids = self.qwen2_5.generate(**inputs, max_new_tokens=128)


        # TODO阶段一 注意力增强 这里看看怎么取出图片id
        if self.pai_enable_attn:
            img_start_idx, img_end_idx = self._resolve_img_token_span(messages, inputs)
            self._inject_qwen_pai_attention(img_start_idx, img_end_idx)

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

        generated_ids = self.qwen2_5.generate(
            **inputs,
            max_new_tokens=256,
            logits_processor=logits_processors
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
        reply = self.__call_qwen2_5(raw_image, attr_prompt)
        trimmed_reply = trim_answer(reply)
        return reply, trimmed_reply

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

        outputs = self.qwen2_5.language_model.generate(
            inputs_embeds=prompts_embeds,
            attention_mask=attention_mask)

        outputs = self.qwen2_5_processor.decode(outputs[0], skip_special_tokens=True)
        return outputs
