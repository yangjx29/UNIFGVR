import os
import sys
import numpy as np
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel
from sklearn.metrics.pairwise import cosine_similarity

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cvd.cdv_captioner import CDVCaptioner
from agents.mllm_bot import MLLMBot
import json
import base64

class MultimodalRetrieval:
    def __init__(self, image_encoder_name="/home/Dataset/Models/Clip/clip-vit-base-patch32", text_encoder_name="/home/Dataset/Models/Clip/clip-vit-base-patch32", fusion_method="concat", device="cuda" if torch.cuda.is_available() else "cpu"):
        """
        Initialize the Multimodal Retrieval module.
        
        Args:
            image_encoder_name (str): Name of the model for image encoding (e.g., CLIP).
            text_encoder_name (str): Name of the model for text encoding (e.g., CLIP).
            fusion_method (str): Method to fuse image and text features ('concat' or 'average').
            device (str): Device to run models on ('cuda' or 'cpu').
        """
        self.device = device
        self.fusion_method = fusion_method
        
        # Load CLIP for image and text feature extraction (CLIP can handle both)
        self.clip_model = CLIPModel.from_pretrained(image_encoder_name).to(self.device)
        self.clip_processor = CLIPProcessor.from_pretrained(image_encoder_name)
        
        # If text encoder is different, load separately; here assuming same as image for CLIP

    def extract_image_feat(self, image_path):
        """
        Extract image features using the encoder.
        
        Args:
            image_path (str): Path to the image file.
        
        Returns:
            np.ndarray: Image feature vector.
        """
        image = Image.open(image_path).convert("RGB")
        inputs = self.clip_processor(images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            feat = self.clip_model.get_image_features(**inputs).cpu().numpy()
        feat = feat.flatten()
        # L2 normalize
        norm = np.linalg.norm(feat) + 1e-12
        feat = feat / norm
        return feat  # 1D

    def extract_text_feat(self, text):
        """
        Extract text features using the encoder.
        
        Args:
            text (str): Text description.
        
        Returns:
            np.ndarray: Text feature vector.
        """
        inputs = self.clip_processor(text=[text], return_tensors="pt", padding=True, truncation=True).to(self.device)
        with torch.no_grad():
            feat = self.clip_model.get_text_features(**inputs).cpu().numpy()
        feat = feat.flatten()
        # L2 normalize
        norm = np.linalg.norm(feat) + 1e-12
        feat = feat / norm
        return feat  # 1D

    def fuse_features(self, img_feat, text_feat):
        """
        Fuse image and text features.
        
        Args:
            img_feat (np.ndarray): Image feature.
            text_feat (np.ndarray): Text feature.
        
        Returns:
            np.ndarray: Fused multimodal feature.
        """
        if self.fusion_method == "concat":
            return np.concatenate([img_feat, text_feat])
        elif self.fusion_method == "average":
            return (img_feat + text_feat) / 2
        else:
            raise ValueError("Invalid fusion method. Use 'concat' or 'average'.")

    def l2_normalize(self, x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
        """L2 归一化到单位范数。"""
        norm = np.linalg.norm(x) + eps
        return x / norm

    def load_gallery_from_json(self,load_path):
        """
        从 JSON 加载 gallery。
        
        Args:
            load_path (str): 加载路径
            compressed (bool): 是否压缩过的
        
        Returns:
            dict: {category: np.array(feature)}
        """
        with open(load_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"Type of gallary: {type(data)}, gallary keys: {list(data.keys()) if isinstance(data, dict) else 'Not a dict'}")
        
        # 如果data是列表，取第一个元素；如果是字典，直接使用
        if isinstance(data, list):
            data = data[0]
        
        gallery = {}
        for cat, value in data.items():
            # 从 list 转为 array
            arr = np.array(value, dtype=np.float32)
            gallery[cat] = arr
        
        print(f'gallery:{gallery}')
        return gallery

    def topk_search(self, query_feat: np.ndarray, gallery_feats: np.ndarray, gallery_cats, k: int = 1):
        """
        用融合特征做 Top-K 相似度检索。
        输入：
            query_feat: np.ndarray [D]
            gallery_feats: np.ndarray [N, D]
            gallery_cats: List[str]
        返回：
            indices: np.ndarray [k]
            sims: np.ndarray [k]
            cats_topk: List[str] [k]
        """
        if gallery_feats.ndim != 2:
            raise ValueError("gallery_feats must be 2D [N, D]")
        q = self.l2_normalize(query_feat.astype(np.float32))
        G = gallery_feats  # 已经在加载时做过归一化
        sims = G @ q  # 余弦相似度（单位向量点积）
        k = min(k, sims.shape[0])
        idx = np.argsort(-sims)[:k]
        return idx, sims[idx], [gallery_cats[i] for i in idx]

    def build_template_gallery(self, mllm_bot, train_samples, cdv_captioner, superclass, kshot=5,region_num=3):
        """
        Build the multimodal category template database (gallery).
        
        Args:
            train_samples (dict): {category: [image_paths]} for few-shot training samples.
            cdv_captioner (CDVCaptioner): Instance of CDVCaptioner to generate descriptions.
            superclass (str): Superclass for captioning (e.g., 'dog').
        
        Returns:
            dict: {category: multimodal_template_feat} where template is average of K-shot fused features.
        """
        gallery = {}
        
        for cat, paths in train_samples.items():
            cat_feats = []
            for i, path in enumerate(paths):
                # Generate description using CDV-Captioner
                description = cdv_captioner.generate_description(mllm_bot, path, train_samples, superclass, kshot, region_num, label=cat, label_id=i)
                
                # Extract features
                img_feat = self.extract_image_feat(path)
                text_feat = self.extract_text_feat(description)
                
                # Fuse
                fused_feat = self.fuse_features(img_feat, text_feat)
                cat_feats.append(fused_feat)
            
            # Average features for the category template
            if cat_feats: 
                gallery[cat] = np.mean(cat_feats, axis=0) # TODO 按行还是列取平均
            else:
                raise ValueError(f"No features extracted for category {cat}")
            print(f'种类:{cat}, 对应的gallery[cat]:{gallery[cat]}')
        return gallery # gallery: {cat: multimodal_template_feat}

    # def query_retrieval(self, mllm_bot, query_image_path, gallery, cdv_captioner, superclass, train_samples):
    #     """
    #     Perform retrieval for a query image (for validation purposes).
        
    #     Args:
    #         query_image_path (str): Path to the query image.
    #         gallery (dict): The built template gallery {cat: template_feat}.
    #         cdv_captioner (CDVCaptioner): Instance for generating query description.
    #         superclass (str): Superclass.
        
    #     Returns:
    #         str: Predicted category (nearest template).
    #     """
    #     # Generate description for query (use real train_samples for reference selection)
    #     description = cdv_captioner.generate_description_inference(mllm_bot, query_image_path, train_samples, superclass)
        
    #     # Extract and fuse query features
    #     img_feat = self.extract_image_feat(query_image_path)
    #     text_feat = self.extract_text_feat(description)
    #     query_feat = self.fuse_features(img_feat, text_feat)
        
    #     # Compute similarities to gallery templates (dot equals cosine since vectors normalized)
    #     similarities = {cat: float(np.dot(query_feat, template)) for cat, template in gallery.items()}
        
    #     # Find the category with highest similarity
    #     predicted_cat = max(similarities, key=similarities.get)
        
        return predicted_cat

    def fgvc_via_multimodal_retrieval(self, mllm_bot, query_image_path, gallery, cdv_captioner, superclass):
        """
        Perform FGVC via multimodal retrieval for a single query image.

        Args:
            query_image_path (str): Path to the query image.
            gallery (dict): The built template gallery {cat: template_feat}.
            cdv_captioner (CDVCaptioner): Instance for generating query description.
            superclass (str): Superclass.

        Returns:
            tuple: (predicted_category, affinity_scores) where affinity_scores is {category: score}.
        """
        # Generate description for query using CDV-Captioner
        description = cdv_captioner.generate_description_inference(mllm_bot,query_image_path, superclass)
        
        # Extract and fuse query features
        img_feat = self.extract_image_feat(query_image_path)
        text_feat = self.extract_text_feat(description)
        query_feat = self.fuse_features(img_feat, text_feat)
        # Normalize query feature
        # query_feat = self.normalize_feat(query_feat)
        
        # Prepare gallery features as matrix (C, dim)
        gallery_cats = list(gallery.keys())
        gallery_feats = np.array([gallery[cat] for cat in gallery_cats])  # Shape: (C, dim)
        print(f'构造的gallery_feats矩阵: {gallery_feats}\nshape:{gallery_feats.shape}')
        # Compute cosine similarities: Fquery * F_gallery^T (1, C)
        cos_sims = np.dot(query_feat, gallery_feats.T)  # Shape: (C,)
        print(f'cos_sims shape: {cos_sims.shape}, cos_sims: {cos_sims}')
        
        # Compute affinities R = exp(-β (1 - cos_sims))
        # TODO 论文没有给值
        beta = 0.5
        affinities = np.exp(-beta * (1 - cos_sims))  # Shape: (C,)
        print(f'affinities shape: {affinities.shape}, affinities: {affinities}')
        
        # Create dict of affinity scores
        affinity_scores = {gallery_cats[i]: affinities[i] for i in range(len(gallery_cats))}
        
        # Predict the category with the highest affinity
        predicted_category = max(affinity_scores, key=affinity_scores.get)
        print(f'predicted_category:{predicted_category}, affinity_scores:{affinity_scores}')
        return predicted_category, affinity_scores

    def evaluate_fgvc(self, mllm_bot, test_samples, gallery, cdv_captioner, superclass):
        """
        Evaluate FGVC on a set of test samples.

        Args:
            test_samples (dict): {true_category: [image_paths]} for test images.
            gallery (dict): The built template gallery.
            cdv_captioner (CDVCaptioner): Instance for generating descriptions.
            superclass (str): Superclass.

        Returns:
            float: Accuracy (correct predictions / total images).
        """
        correct = 0
        total = 0
        
        for true_cat, paths in test_samples.items():
            for path in paths:
                predicted_cat, _ = self.fgvc_via_multimodal_retrieval(mllm_bot, path, gallery, cdv_captioner, superclass)
                if predicted_cat == true_cat or predicted_cat in true_cat or true_cat in predicted_cat:
                    correct += 1
                total += 1
        
        accuracy = correct / total if total > 0 else 0.0
        return accuracy

# Example usage (for testing)
if __name__ == "__main__":
    """
    CUDA_VISIBLE_DEVICES=2 python multimodal_retrieval.py 2>&1 | tee ../logs/interence_dog.log
    """
    # Initialize modules
    captioner = CDVCaptioner()
    retrieval = MultimodalRetrieval()
    mllm_bot = MLLMBot(model_tag="Qwen2.5-VL-8B", model_name="Qwen2.5-VL-8B", device="cuda" if torch.cuda.is_available() else "cpu")
    
    # Dummy train_samples 
    # train_samples = {
    #     "cat1": ["path/to/img1.jpg", "path/to/img2.jpg"],
    #     "cat2": ["path/to/img3.jpg", "path/to/img4.jpg"]
    # }
    
    # # 构建检索库
    # gallery = retrieval.build_template_gallery(mllm_bot, train_samples, captioner, superclass="dog")
    # print("Gallery built with categories:", list(gallery.keys()))
    
    # Test query retrieval
    # query_path = "/data/yjx/MLLM/UniFGVR/datasets/dogs_120/Images/n02085620-Chihuahua/n02085620_1558.jpg" #chihuahua
    # gallery = retrieval.load_gallery_from_json('/data/yjx/MLLM/UniFGVR/experiments/dog120/gallery/dog120_gallery.json') 
    # predicted = retrieval.fgvc_via_multimodal_retrieval(mllm_bot, query_path, gallery, captioner, "dog")
    # print(f"Predicted category for query: {predicted}")
    # query_path = "/data/yjx/MLLM/UniFGVR/datasets/dogs_120/Images/n02085620-Chihuahua/n02085620_1558.jpg" #chihuahua
    gallery = retrieval.load_gallery_from_json('/data/yjx/MLLM/UniFGVR/experiments/dog120/gallery/dog120_gallery.json') 
    test_samples = {}
    # 构建test samples
    img_root = "/data/yjx/MLLM/UniFGVR/datasets/dogs_120/Images"
    class_folders = os.listdir(img_root)
    for i in range(len(class_folders)):
        cat_name = class_folders[i].split('-')[-1].replace('_', ' ')
        # print(f'cat name:{cat_name}')
        img_path = os.path.join(img_root, class_folders[i])
        file_names = os.listdir(img_path)
        # print(f'img_path:{img_path}\tfilename:{file_names}')
        for name in file_names:
            path = os.path.join(img_path,name)
            if cat_name not in test_samples:
                test_samples[cat_name] = []
            test_samples[cat_name].append(path)

    # print(f'test sample:{test_samples}')
    accuracy = retrieval.evaluate_fgvc(mllm_bot, test_samples, gallery, captioner, "dog")
    print(f"accuracy: {accuracy}")