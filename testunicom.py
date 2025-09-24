# # pip install git+https://github.com/huggingface/transformers@v4.51.3-MLCD-preview

# import requests
# from PIL import Image
# from transformers import AutoProcessor, MLCDVisionModel

# import torch

# # Load model and processor
# model = MLCDVisionModel.from_pretrained("/home/Dataset/Models/mlcd/mlcd-vit-bigG-patch14-448/unicom")
# processor = AutoProcessor.from_pretrained("/home/Dataset/Models/mlcd/mlcd-vit-bigG-patch14-448/unicom")

# # Process single image
# target_image_path='/data/yjx/MLLM/UniFGVR/datasets/dogs_120/images_discovery_all_3/000.Chihuaha/000.Chihuaha_n02085620_7613.jpg'
# image = Image.open(target_image_path).convert("RGB")
# inputs = processor(images=image, return_tensors="pt")

# # Generate outputs
# with torch.no_grad():
#     outputs = model(**inputs)

# # Get visual features
# features = outputs.last_hidden_state

# print(f"Extracted features shape: {features.shape}")
from difflib import SequenceMatcher
def is_similar(str1, str2, threshold=0.7):
        """判断两个字符串是否语义相似"""
        
        similarity = SequenceMatcher(None, str1, str2).ratio()
        return similarity >= threshold

prediction=' Shih-Tzu'
predicted_category='Shih-Tzu'
print(is_similar(prediction,predicted_category))