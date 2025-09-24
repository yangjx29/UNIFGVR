from transformers import AutoModel, AutoTokenizer, AutoImageProcessor
import torch
from PIL import Image
import requests
from io import BytesIO
import os


# 模型路径
model_path = "/home/Dataset/Models/Clip/jina-clip-v2"

# !pip install transformers einops timm pillow
from modelscope import AutoModel

# Initialize the model
model = AutoModel.from_pretrained(model_path, trust_remote_code=True)

# Corpus
sentences = [
    'غروب جميل على الشاطئ', # Arabic
    '海滩上美丽的日落', # Chinese
    'Un beau coucher de soleil sur la plage', # French
    'Ein wunderschöner Sonnenuntergang am Strand', # German
    'Ένα όμορφο ηλιοβασίλεμα πάνω από την παραλία', # Greek
    'समुद्र तट पर एक खूबसूरत सूर्यास्त', # Hindi
    'Un bellissimo tramonto sulla spiaggia', # Italian
    '浜辺に沈む美しい夕日', # Japanese
    '해변 위로 아름다운 일몰', # Korean
]

# Public image URLs or PIL Images
image_urls = ['https://i.ibb.co/nQNGqL0/beach1.jpg', 'https://i.ibb.co/r5w8hG8/beach2.jpg']

# Choose a matryoshka dimension, set to None to get the full 1024-dim vectors
truncate_dim = 512

# Encode text and images
text_embeddings = model.encode_text(sentences, truncate_dim=truncate_dim)
image_embeddings = model.encode_image(
    image_urls, truncate_dim=truncate_dim
)  # also accepts PIL.Image.Image, local filenames, dataURI

# Encode query text
query = 'beautiful sunset over the beach' # English
query_embeddings = model.encode_text(
    query, task='retrieval.query', truncate_dim=truncate_dim
)

# Text to Image
print('En -> Img: ' + str(query_embeddings @ image_embeddings[0].T))
# Image to Image
print('Img -> Img: ' + str(image_embeddings[0] @ image_embeddings[1].T))
# Text to Text
print('En -> Ar: ' + str(query_embeddings @ text_embeddings[0].T))
print('En -> Zh: ' + str(query_embeddings @ text_embeddings[1].T))
print('En -> Fr: ' + str(query_embeddings @ text_embeddings[2].T))
print('En -> De: ' + str(query_embeddings @ text_embeddings[3].T))
print('En -> Gr: ' + str(query_embeddings @ text_embeddings[4].T))
print('En -> Hi: ' + str(query_embeddings @ text_embeddings[5].T))
print('En -> It: ' + str(query_embeddings @ text_embeddings[6].T))
print('En -> Jp: ' + str(query_embeddings @ text_embeddings[7].T))
print('En -> Ko: ' + str(query_embeddings @ text_embeddings[8].T))
"""
CUDA_VISIBLE_DEVICES=1 python
"""