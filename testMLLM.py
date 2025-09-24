from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
from PIL import Image
from agents.mllm_bot import MLLMBot

target_image_path='/data/yjx/MLLM/UniFGVR/datasets/dogs_120/images_discovery_all_7/027.Weimaraner/027.Weimaraner_n02092339_4003.jpg'

prompt='Describe the most significant features that can identify the type of dog in the image, and ultimately tell me what kind of dog it is.'

model = MLLMBot(model_tag='Qwen2.5-VL-7B', model_name='Qwen2.5-VL-7B', pai_enable_attn=False,device='cuda', device_id='0', bit8=False) 

image=Image.open(target_image_path).convert("RGB")
reply, output_text = model.describe_attribute(image, prompt)
print(f'output_text: {output_text}')
"""
CUDA_VISIBLE_DEVICES=1 python testMLLM.py | tee ./logs/testqwen_model.log
"""