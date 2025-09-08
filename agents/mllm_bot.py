import torch
from os import path
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
from PIL import Image

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
        generated_ids = self.qwen2_5.generate(**inputs, max_new_tokens=128)
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
