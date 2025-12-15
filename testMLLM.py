from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
from PIL import Image
from agents.mllm_bot import MLLMBot

target_image_path='/data/yjx/MLLM/UniFGVR/datasets/dogs_120/images_discovery_all_7/027.Weimaraner/027.Weimaraner_n02092339_4003.jpg'

prompt='Describe the most important features in the image that can identify the type of dog (intrinsic features such as nose, eyes, fur, ears, etc.) and ultimately tell me what type of dog it is.'

prompt_forfun = 'Greyhound和Whippet分别是什么品种'
model = MLLMBot(model_tag='Qwen2.5-VL-7B', model_name='Qwen2.5-VL-7B', pai_enable_attn=True,device='cuda', device_id='0', bit8=False) 

processor = model.qwen2_5_processor
messages = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": prompt_forfun},
        ],
    }
]
text = processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
image_inputs, video_inputs = process_vision_info(messages)
inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt",
)
inputs = inputs.to("cuda")

# Inference: Generation of the output
# generated_ids = model.qwen2_5.generate(**inputs, max_new_tokens=512)
# generated_ids_trimmed = [ 
#     out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
# ]
# output_text = processor.batch_decode(
#     generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
# )
# print(f'response: {output_text}')

image=Image.open(target_image_path).convert("RGB")
reply, output_text = model.describe_attribute(image, prompt)
print(f'output_text: {output_text}')
"""
CUDA_VISIBLE_DEVICES=1 python testMLLM.py | tee ./logs/testqwen_model.log
"""