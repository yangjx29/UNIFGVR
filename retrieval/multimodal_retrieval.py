import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# 直接导入UniFGVR目录下的utils模块
sys.path.insert(0, '/data/yjx/MLLM/UniFGVR')
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from transformers import CLIPProcessor, CLIPModel
from transformers import Blip2Processor, Blip2Model
from transformers import AutoProcessor, BlipForImageTextRetrieval
from sklearn.metrics.pairwise import cosine_similarity

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.util import is_similar
from cvd.cdv_captioner import CDVCaptioner
from agents.mllm_bot import MLLMBot
import json
import base64
from torch.nn import functional as F

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
        self.blip_processor = AutoProcessor.from_pretrained("Salesforce/blip-itm-base-coco")
        self.blip_model = BlipForImageTextRetrieval.from_pretrained("Salesforce/blip-itm-base-coco")
        # 确保模型与输入在同一设备，避免 FloatTensor/CudaFloatTensor 不一致
        self.blip_model = self.blip_model.to(self.device)
        self.blip_model.eval()

        # # Lazy-initialized cross-attention components
        # self._cross_d_model: int = 1024
        # self._cross_n_heads: int = 8
        # self._proj_img: nn.Linear | None = None
        # self._proj_txt: nn.Linear | None = None
        # self._attn: nn.MultiheadAttention | None = None
        # self._proj_out: nn.Linear | None = None
        
        # If text encoder is different, load separately; here assuming same as image for CLIP
        # Pretrained cross-modal attention (BLIP) - lazy
        # self._blip_processor: Blip2Processor | None = None
        # self._blip_model: Blip2Model | None = None

    def init_blip(self):
        # self._blip_processor = Blip2Processor.from_pretrained("/home/Dataset/Models/blip/blip2-opt-6.7b-coco")
        # self._blip_model = Blip2Model.from_pretrained("/home/Dataset/Models/blip/blip2-opt-6.7b-coco").to(self.device)
        self.blip_processor = AutoProcessor.from_pretrained("Salesforce/blip-itm-base-coco")
        self.blip_model = BlipForImageTextRetrieval.from_pretrained("Salesforce/blip-itm-base-coco")
        # 确保模型与输入在同一设备，避免 FloatTensor/CudaFloatTensor 不一致
        self.blip_model = self.blip_model.to(self.device)
        self.blip_model.eval()

    def extract_multimodal_feat_blip(self, image_path: str, text: str):
        # self.init_blip()
        img = Image.open(image_path).convert("RGB")
        inputs = self.blip_processor(images=img, text=text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.blip_model(**inputs, output_attentions=False, output_hidden_states=True)
            print(f'outputs.keys():{outputs.keys()}')
            if hasattr(outputs, 'text_last_hidden_state') and outputs.text_last_hidden_state is not None:
                fused = outputs.text_last_hidden_state[:, 0, :]  # (1, D)
            else:
                fused = outputs.question_embeds 
        print(f'fused feature shape:{fused.shape}')
        fused = fused.squeeze(0)
        fused = fused.mean(dim=0) 
        fused = fused / (fused.norm(p=2) + 1e-12)
        return fused.detach().cpu().numpy()

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

    # TODO 特征融合 考虑cross-attention
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
        elif self.fusion_method == "weighted":
            alpha = 0.7
            return alpha * text_feat + (1 - alpha) * img_feat
        elif self.fusion_method == "cross_atten":
            # 已处理
            raise RuntimeError("blip_cross requires raw image path and text; call extract_multimodal_feat_blip().")
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
                
                # Extract fused features (use BLIP if requested)
                if self.fusion_method == "cross_atten":
                    fused_feat = self.extract_multimodal_feat_blip(path, description)
                    print(f'after extract_multimodal_feat_blip fused_feat shape:{fused_feat.shape}')
                else:
                    img_feat = self.extract_image_feat(path)
                    text_feat = self.extract_text_feat(description)
                    fused_feat = self.fuse_features(img_feat, text_feat)
                # fused_feat = F.normalize(fused_feat)
                cat_feats.append(fused_feat)
            
            # Average features for the category template
            if cat_feats: 
                gallery[cat] = np.mean(cat_feats, axis=0) # TODO 按行还是列取平均
            else:
                raise ValueError(f"No features extracted for category {cat}")
            # print(f'种类:{cat}, 对应的gallery[cat]:{gallery[cat]}')
            print(f'种类:{cat}')
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

    def fgvc_via_multimodal_retrieval(self, mllm_bot, query_image_path, gallery, cdv_captioner, superclass, use_rag=True,topk=1):
        """
        Perform FGVC via multimodal retrieval for a single query image.

        Args:
            query_image_path (str): Path to the query image.
            gallery (dict): The built template gallery {cat: template_feat}.
            cdv_captioner (CDVCaptioner): Instance for generating query description.
            superclass (str): Superclass.
            use_rag (bool): Whether to use RAG with Top-5 candidates (True) or simple Top-1 (False).

        Returns:
            tuple: (predicted_category, affinity_scores) where affinity_scores is {category: score}.
        """
        # Generate description for query using CDV-Captioner
        description = cdv_captioner.generate_description_inference(mllm_bot,query_image_path, superclass)
        
        # Extract and fuse query features
        if self.fusion_method == "cross_atten":
            query_feat = self.extract_multimodal_feat_blip(query_image_path, description)
        else:
            img_feat = self.extract_image_feat(query_image_path)
            text_feat = self.extract_text_feat(description)
            query_feat = self.fuse_features(img_feat, text_feat)
        # 归一化
        # query_feat = F.normalize(query_feat) 
        # Normalize query feature
        # query_feat = self.normalize_feat(query_feat)
        
        # Prepare gallery features as matrix (C, dim)
        gallery_cats = list(gallery.keys())
        gallery_feats = np.array([gallery[cat] for cat in gallery_cats])  # Shape: (C, dim)
        # Ensure dims align
        if query_feat.shape[-1] != gallery_feats.shape[-1]:
            raise ValueError(f"Dim mismatch: query {query_feat.shape[-1]} vs gallery {gallery_feats.shape[-1]}. Ensure same fusion_method for gallery and query.")
        # 归一化
        # gallery_feats = F.normalize(gallery_feats)
        print(f'构造的gallery_feats矩阵shape:{gallery_feats.shape}')
        # Compute cosine similarities: Fquery * F_gallery^T (1, C)
        cos_sims = np.dot(query_feat, gallery_feats.T)  # Shape: (C,)
        # print(f'cos_sims shape: {cos_sims.shape}, cos_sims: {cos_sims}')
        
        # Compute affinities R = exp(-β (1 - cos_sims))
        # TODO 论文没有给值
        beta = 0.1
        affinities = np.exp(-beta * (1 - cos_sims))  # Shape: (C,)
        # print(f'affinities shape: {affinities.shape}, affinities: {affinities}')
        
        # Create dict of affinity scores
        affinity_scores = {gallery_cats[i]: affinities[i] for i in range(len(gallery_cats))}
        # predicted_category = max(affinity_scores, key=affinity_scores.get)
        if use_rag:
            # 使用RAG：获取Top-5候选类别，然后让MLLM进行最终推理
            topk_categories = sorted(affinity_scores.items(), key=lambda x: x[1], reverse=True)[:topk]
            topk_cat_names = [cat for cat, score in topk_categories]
            topk_scores = [score for cat, score in topk_categories]
            
            print(f"Top-{topk} candidates: {topk_cat_names}")
            print(f"Top-{topk} scores: {topk_scores}")
            
            # 构造RAG prompt让MLLM进行最终推理
            rag_prompt = self._construct_rag_prompt(topk_cat_names, topk_scores, superclass)
            
            # 调用MLLM进行最终推理（需要传入图像）
            # 先加载图像
            from PIL import Image
            query_image = Image.open(query_image_path).convert("RGB")
            reply, final_prediction = mllm_bot.describe_attribute(query_image, rag_prompt)  # 使用describe_attribute获取清理后的输出
            print(f'final_prediction:{final_prediction}')
            # 从MLLM输出中提取最终预测类别
            predicted_category = self._extract_final_category(final_prediction, topk_cat_names)
            print(f"Final prediction after RAG: {predicted_category}")
        else:
            # 使用简单的Top-1方法
            predicted_category = max(affinity_scores, key=affinity_scores.get)
            print(f"Top-1 prediction: {predicted_category}")
        
        print(f'predicted_category:{predicted_category}')
        return predicted_category, affinity_scores
    def evaluate_fgvc(self, mllm_bot, test_samples, gallery, cdv_captioner, superclass, use_rag=True, topk = 1):
        """
        Evaluate FGVC on a set of test samples.

        Args:
            test_samples (dict): {true_category: [image_paths]} for test images.
            gallery (dict): The built template gallery.
            cdv_captioner (CDVCaptioner): Instance for generating descriptions.
            superclass (str): Superclass.
            use_rag (bool): Whether to use RAG with Top-k candidates (True) or simple Top-1 (False).

        Returns:
            float: Accuracy (correct predictions / total images).
        """
        correct = 0
        total = 0
        
        for true_cat, paths in test_samples.items():
            for path in paths:
                predicted_cat, _ = self.fgvc_via_multimodal_retrieval(mllm_bot, path, gallery, cdv_captioner, superclass, use_rag,topk)
                # if predicted_cat == true_cat or predicted_cat in true_cat or true_cat in predicted_cat:
                #     correct += 1
                #     print(f'匹配成功,correct:{correct},total:{total}')
                if is_similar(predicted_cat,true_cat):
                    correct += 1
                    print(f'匹配成功,correct:{correct},total:{total}')
                total += 1
                print(f'succ:{correct / total if total > 0 else 0.0}')
                print(f'predicted_cat:{predicted_cat}, true_cat:{true_cat}')
        
        accuracy = correct / total if total > 0 else 0.0
        return accuracy

    def _construct_rag_prompt(self, top5_categories, top5_scores, superclass):
        """
        构造RAG prompt，让MLLM基于Top-5候选类别进行最终推理。
        
        Args:
            query_image_path (str): 查询图像路径
            top5_categories (list): Top-5候选类别名称
            top5_scores (list): 对应的相似度分数
            superclass (str): 超类名称
            
        Returns:
            str: 构造好的RAG prompt
        """
        # 构造候选类别列表
        candidates_text = ""
        for i, (cat, score) in enumerate(zip(top5_categories, top5_scores)):
            candidates_text += f"{i+1}. {cat} (scores: {score:.4f})\n"
        
        rag_prompt = f"""<|im_start|>system
            You are an expert in fine-grained visual recognition. You will be given an image and the top-5 most similar categories from a {superclass} dataset based on multimodal features. Your task is to analyze the image carefully and select the most accurate category from the candidates.

            Please consider:
            1. Visual characteristics specific to each breed/variety
            2. Distinguishing features that differentiate between similar categories
            3. Overall appearance, proportions, and distinctive traits

            Respond with ONLY the category name that best matches the image.<|im_end|>

            <|im_start|>user
            <image>
            Based on the multimodal similarity analysis, here are the top-5 candidate categories:

            {candidates_text}

            Please analyze this image and select the most accurate category from the above candidates. Consider the visual characteristics and distinguishing features of each breed/variety.

            Respond with ONLY the category name (e.g., "Chihuahua" or "Shiba Inu").<|im_end|>

            <|im_start|>assistant"""
        
        return rag_prompt

    def _extract_final_category(self, mllm_output, top5_categories):
        """
        从MLLM输出中提取最终预测的类别名称。
        
        Args:
            mllm_output (str): MLLM的原始输出
            top5_categories (list): Top-5候选类别列表
            
        Returns:
            str: 提取出的最终预测类别
        """
        # 清理输出，去除多余空白和标点
        cleaned_output = mllm_output[0].strip().replace('.', '').replace(',', '').replace('-', '').replace('.', '')
        
        # 尝试直接匹配候选类别
        for category in top5_categories:
            if category.lower() in cleaned_output.lower():
                return category
        
        # 如果直接匹配失败，尝试模糊匹配
        for category in top5_categories:
            # 检查是否包含类别的主要部分
            category_words = category.lower().split()
            if any(word in cleaned_output.lower() for word in category_words if len(word) > 2):
                return category
        
        # 如果都匹配失败，返回Top-1候选（作为fallback）
        print(f"Warning: Could not extract category from MLLM output: '{mllm_output}'")
        print(f"Using top-1 candidate as fallback: {top5_categories[0]}")
        return top5_categories[0]

# Example usage (for testing)
if __name__ == "__main__":
    """
    CUDA_VISIBLE_DEVICES=1 python multimodal_retrieval.py 2>&1 | tee ../logs/interence1_dog_rag_tok5_concat_add_token_weighted_10_28layer_image_qufen0.2_duibi_newquanzhong.log
    CUDA_VISIBLE_DEVICES=0 nohup python multimodal_retrieval.py 2>&1 | tee ../logs/interence1_dog_rag_tok5_concat_add_token_weighted_10_28layer_image_qufen.log &
    CUDA_VISIBLE_DEVICES=0 python multimodal_retrieval.py 2>&1 | tee ../logs/interence1_dog_mllm_direct_withatten_after.log
    """
    # Initialize modules
    fusion_method="concat"
    captioner = CDVCaptioner()
    retrieval = MultimodalRetrieval(image_encoder_name="/home/Dataset/Models/Clip/clip-vit-base-patch32", text_encoder_name="/home/Dataset/Models/Clip/clip-vit-base-patch32", fusion_method=fusion_method, device="cuda" if torch.cuda.is_available() else "cpu")
    mllm_bot = MLLMBot(model_tag="Qwen2.5-VL-7B", model_name="Qwen2.5-VL-7B", pai_enable_attn=True, device="cuda" if torch.cuda.is_available() else "cpu")
    
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
    gallery = retrieval.load_gallery_from_json('/data/yjx/MLLM/UniFGVR/experiments/dog120/gallery/dog120_gallery_concat_old.json') 
    test_samples = {}
    # 构建test samples
    # img_root = "/data/yjx/MLLM/UniFGVR/datasets/dogs_120/Images"
    img_root = "/data/yjx/MLLM/UniFGVR/datasets/dogs_120/images_discovery_all_1"
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

    print(f'test sample:{test_samples}')
    # 使用RAG进行Top-5候选推理
    accuracy_rag = retrieval.evaluate_fgvc(mllm_bot, test_samples, gallery, captioner, "dog", use_rag=True, topk=5)
    print(f"accuracy with RAG: {accuracy_rag}")
    
    # 对比：使用简单Top-1方法
    accuracy_top1 = retrieval.evaluate_fgvc(mllm_bot, test_samples, gallery, captioner, "dog", use_rag=False)
    print(f"accuracy with Top-1: {accuracy_top1}")
    # 直接用MLLM
    # superclass = 'dog'
    # from data import DATA_STATS, PROMPTERS, DATA_DISCOVERY  
    # cname_sheet = DATA_STATS[superclass]['class_names']
    # prompt=f'Look carefully at the dog in the picture and tell me which category it belongs to in {cname_sheet}. The answer only needs the specific category name of {cname_sheet}, no extra explanation is needed.'
    
    # total=0
    # accuracy_dirct=0
    # correct=0
    # for true_cat, paths in test_samples.items():
    #     for path in paths:
    #         image = Image.open(path).convert("RGB")
    #         _, predicted_cat = mllm_bot.describe_attribute(image, prompt)
    #         true_cat=true_cat.split('.')[1]
    #         predicted_cat=predicted_cat[0]
    #         if is_similar(predicted_cat,true_cat, threshold=0.5):
    #             correct += 1
    #             print(f'匹配成功,correct:{correct},total:{total}')
    #         total += 1
    #         print(f'succ:{correct / total if total > 0 else 0.0}')
    #         print(f'predicted_cat:{predicted_cat}, true_cat:{true_cat}')
        
    #     accuracy_dirct = correct / total if total > 0 else 0.0
    # print(f'accuracy with direct:{accuracy_dirct}')