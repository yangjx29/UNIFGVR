"""
该函数用于构建检索模板库

我们首先提出了类别区分视觉描述子( Category-Discriminative Visual Captioner，CDV-Captioner )，它是一个结构化描述模块，使用多模态虽然链提示来指导MLLMs自适应地识别目标图像中的关键区分区域。然后将这些区域翻译成细粒度的结构化文本描述，以捕获FGVC所必需的细微的视觉差异。基于这些描述，每个图像被转换成图像-文本对，形成丰富的多模态表示，用于下游检索。然后，我们使用任何现成的预训练编码器提取混合视觉-语言特征，并通过计算多模态类别模板数据库的特征相似度来进行细粒度的类别识别。

take dog dateset as example, query MLLM:
1. Discriminative Region Discovery prompt: We provide t images from different categories within the dog that share similar visual features, and use them as references to generate s discriminative visual regions for distinguishing the target image's category.
2. Region Attribute Description prompt: Describe the visual attributes of the white fur / upright ear / softly angular face in the dog category.
3. Attribute Feature Summarization prompt: Summarize the information you get about the dog from the attribute description.
"""

import os
import numpy as np
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel
from agents.mllm_bot import MLLMBot
from sklearn.metrics.pairwise import cosine_similarity
import random
import re
import json
from utils.fileios import dump_json, load_json, dump_txt 
from utils.util import *
class CDVCaptioner:
    def __init__(self, image_encoder_name="/home/Dataset/Models/Clip/clip-vit-base-patch32", device="cuda" if torch.cuda.is_available() else "cpu", cfg = None):
        """
        Initialize the CDV-Captioner module.
        
        Args:
            image_encoder_name (str): Name of the CLIP model for image encoding.
            device (str): Device to run models on ('cuda' or 'cpu').
        """
        self.device = device
        self.cfg = cfg
        # Load CLIP for image feature extraction
        self.clip_model = CLIPModel.from_pretrained(image_encoder_name).to(self.device)
        self.clip_processor = CLIPProcessor.from_pretrained(image_encoder_name)
        
    def extract_image_feat(self, image_path):
        """
        Extract image features using CLIP.
        
        Args:
            image_path (str): Path to the image file.
        
        Returns:
            np.ndarray: Image feature vector.
        """
        image = Image.open(image_path).convert("RGB")
        inputs = self.clip_processor(images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            feat = self.clip_model.get_image_features(**inputs).cpu().numpy()
        # L2 normalize for cosine similarity stability
        # norm = np.linalg.norm(feat, axis=1, keepdims=True) + 1e-12
        # feat = feat / norm
        return feat

    # TODO Reference Sample Selection.
    def select_references(self, target_image_path, train_samples, t=5):
        """
        Select top-t reference images based on visual similarity.
        
        Args:
            target_image_path (str): Path to the target image.
            train_samples (dict): {category: [image_paths]} for few-shot training samples.
            t (int): Number of reference images to select.
        
        Returns:
            list: List of t reference image paths.
        """
        # Compute cluster centers for each category
        cluster_centers = {}  #{category: [image_features_mean]}
        for category, paths in train_samples.items():
            feats = [self.extract_image_feat(p) for p in paths]
            cluster_centers[category] = np.mean(feats, axis=0)
        
        # Extract target feature
        f_t = self.extract_image_feat(target_image_path)
        
        # Compute similarities to cluster centers
        similarities = {category: cosine_similarity(f_t, center)[0][0] for category, center in cluster_centers.items()} # {'name1' : cos1, 'name2 : cos2'}
        
        # Select top-t categories (sorted by similarity descending)
        top_t_cats = sorted(similarities, key=similarities.get, reverse=True)[:t]
        
        # Pick one random image from each top category as reference
        ref_paths = [random.choice(train_samples[category]) for category in top_t_cats]

        save_path_references = self.cfg['path_references'] + str(t)  # 构建LLM提示保存路径
        # run the main program to describe the per-img attributes - 运行主程序描述每张图像的属性
        # list to string
        ref_paths = "\n".join(ref_paths)
        dump_txt(save_path_references, ref_paths)
        print(f"save_path_references: {save_path_references}\n")
        return ref_paths

    # TODO Discriminative Region Discovery.
    def discover_regions(self, mllm_bot, target_image_path, ref_paths, superclass, s=3):
        """
        Discover s discriminative regions using MLLM.
        prompt: {IMAGERY} We provide {t} images from different categories within the {SUPERCLASS} that share similar visual features, and use them as references to generate {s} discriminative visual regions for distinguishing the target image's category.
        Args:
            target_image_path (str): Path to the target image.
            ref_paths (list): List of reference image paths.
            superclass (str): Superclass (e.g., 'dog').
            s (int): Number of regions to generate.
        
        Returns:
            list: List of s region names (e.g., ['white fur', 'upright ears']).
        """
        # string to list
        ref_paths = ref_paths.split("\n")
        t = len(ref_paths)
        print(f'ref_path: {ref_paths}\nlen ref path: {t}')
        
        # Prepare images: target + references
        if t == 1:
            # 测试阶段
            images = [Image.open(target_image_path).convert("RGB")]
        else:
            images = [Image.open(target_image_path).convert("RGB")] + [Image.open(p).convert("RGB") for p in ref_paths]
        
        # Prompt template from paper (formula 1)
        prompt = (
            f"We provide {t} images from different categories within the {superclass} that share similar visual features, "
            f"and use them as references to generate {s} discriminative visual regions(eg,. descriptive phrases like white fur, thin straight legs) for distinguishing the target image's category. "
            "The first image is the target, followed by the references(if have). Output the regions as a comma-separated list."
        )
        
        # Call MLLM (LLaVA supports list of images)
        outputs = []
        
        # 应该是一起输入让qwenqu判断
        reply, trimmed_reply = mllm_bot.describe_attribute(images, prompt)
        print(f'prompt in discover regions:{prompt}\ndiscover_regions: {trimmed_reply}')
        if isinstance(outputs, str):
            trimmed_reply = trimmed_reply.lower().strip()
            outputs = trimmed_reply
            regions = ", ".join(outputs)
        else:
            for target_reference in trimmed_reply:
                output_string = target_reference.lower().strip() 
                feats = output_string.split(",")
                for f in feats:
                    outputs.append(f.strip())
        regions = outputs[:s]  # Take top s
        # save_path_regions = self.cfg['path_regions'] + str(t)
        # dump_json(save_path_regions, regions)
        # print(f"save_path_regions: {save_path_regions}\n")
        return regions # eg,.white fur

    def discover_regions_inference(self, mllm_bot, target_image_path, ref_paths, superclass, s=3):
        """
        Discover s discriminative regions using MLLM.
        prompt: {IMAGERY} We provide {t} images from different categories within the {SUPERCLASS} that share similar visual features, and use them as references to generate {s} discriminative visual regions for distinguishing the target image's category.
        Args:
            target_image_path (str): Path to the target image.
            ref_paths (list): List of reference image paths.
            superclass (str): Superclass (e.g., 'dog').
            s (int): Number of regions to generate.
        
        Returns:
            list: List of s region names (e.g., ['white fur', 'upright ears']).
        """
        # string to list
        ref_paths = ref_paths.split("\n")
        t = len(ref_paths)
        print(f'ref_path: {ref_paths}\nlen ref path: {t}')
        
        # Prepare images: target + references
        if t == 1:
            # 测试阶段
            images = [Image.open(target_image_path).convert("RGB")]
        else:
            images = [Image.open(target_image_path).convert("RGB")] + [Image.open(p).convert("RGB") for p in ref_paths]
        
        # Prompt template from paper (formula 1)
        prompt = (
            f"We provide an image from {superclass}, "
            f"please generate {s} visual regions that are discriminative compared to other subclasses (e.g., descriptive phrases like white fur, thin straight legs) to distinguish the target image's category. Output the regions as a comma-delimited list."
        )
        
        # Call MLLM (LLaVA supports list of images)
        outputs = []
        
        # 应该是一起输入让qwenqu判断
        reply, trimmed_reply = mllm_bot.describe_attribute(images, prompt)
        print(f'prompt in discover regions:{prompt}\ndiscover_regions: {trimmed_reply}')
        if isinstance(outputs, str):
            trimmed_reply = trimmed_reply.lower().strip()
            outputs = trimmed_reply
            regions = ", ".join(outputs)
        else:
            for target_reference in trimmed_reply:
                output_string = target_reference.lower().strip() 
                feats = output_string.split(",")
                for f in feats:
                    outputs.append(f.strip())
        regions = outputs[:s]  # Take top s
        # save_path_regions = self.cfg['path_regions'] + str(t)
        # dump_json(save_path_regions, regions)
        # print(f"save_path_regions: {save_path_regions}\n")
        return regions # eg,.white fur

    # TODO Region Attribute Description.
    def describe_attributes(self, mllm_bot, target_image_path, regions, superclass="dog", label=None, label_id=None):
        """
        Describe attributes for each region using MLLM.
        
        Args:
            mllm_bot: MLLM bot for attribute description.
            target_image_path (str): Path to the target image.
            regions (list): List of region names.
            superclass (str): Superclass.
            label (str, optional): Human-readable label name.
            label_id (int, optional): Label ID.
        
        Returns:
            list: List of attribute descriptions for each region.
        """
        image = Image.open(target_image_path).convert("RGB")
        descriptions = []
        
        for region in regions:    
            if label is not None:
                # 构建阶段
                prompt = f"Given an image, describe the visual attributes of {region} in the given {superclass} category, whose specific category is {label}."
            else:
                prompt = f"Given an image, describe the visual attributes of {region} in the {superclass} category."
            reply, output = mllm_bot.describe_attribute(image, prompt)
            description = ", ".join(output)
            descriptions.append(description)
            print(f'prompt: {prompt} \n description: {description}')
        
        # # 构建JSON格式的结果
        # result_data = {
        #     "img_path": target_image_path,
        #     "label": label if label is not None else "unknown",
        #     "label_id": label_id if label_id is not None else -1,
        #     "regions": regions,
        #     "descriptions": descriptions,
        #     "superclass": superclass
        # }
        # save_path_descriptions = self.cfg['path_descriptions']
        # dump_json(save_path_descriptions, result_data)
        # print(f"Saved descriptions to: {save_path_descriptions}")
        
        return descriptions

    # TODO Attribute Feature Summarization
    def summarize_attributes(self, mllm_bot, target_image_path, descriptions, superclass, regions, label=None, label_id=None):
        """
        Summarize attributes into a structured description using MLLM.
        
        Args:
            target_image_path (str): Path to the target image.
            descriptions (list): List of attribute descriptions.
            superclass (str): Superclass.
            regions (list): List of region names.
        
        Returns:
            str: Final structured description A_i.
        """
        image = Image.open(target_image_path).convert("RGB")
        
        # Combine regions and descriptions for prompt
        attr_info = "\n".join([f"**Region**: {r}\nDescription: {d}" for r, d in zip(regions, descriptions)])
        
        if label is not None:
            # 构建流程
            prompt = (
                f"Summarize the information you get about the {label} from the attribute description:\n{attr_info}\n"
                f"Output a structured description."
            )
        else:
            # 测试流程
            prompt = (
                f"Summarize the information you get about the {superclass} from the attribute description:\n{attr_info}\n"
                "Output a concise structured description."
            )
        reply, output = mllm_bot.describe_attribute(image, prompt)
        output = ", ".join(output)
        output = output.lower()
        if label is not None:
            result_data = {
                "img_path": target_image_path,
                "label": label if label is not None else "unknown",
                "label_id": label_id if label_id is not None else -1,
                "regions": regions,
                "descriptions": descriptions,
                "superclass": superclass
            }
            save_path_descriptions = self.cfg['path_descriptions']
            dump_json(save_path_descriptions, result_data)
            print(f"Saved descriptions to: {save_path_descriptions}")
        print(f'summarize_attributes: {output}')
        return output.strip()


    def generate_description_inference(self, mllm_bot, target_image_path, superclass, t=5, s=3):
        # Step 1: Discover regions
        ref_path = ""
        regions = self.discover_regions_inference(mllm_bot, target_image_path, ref_path, superclass, s)
        
        # Step 2: Describe attributes
        descriptions = self.describe_attributes(mllm_bot, target_image_path, regions, superclass)
        
        # Step 3: Summarize
        final_description = self.summarize_attributes(mllm_bot, target_image_path, descriptions, superclass, regions)
        
        return final_description

    def generate_description(self, mllm_bot, target_image_path, train_samples, superclass, kshot=5, s=3, label=None, label_id=None):
        """
        Main function to generate the structured description for an image.
        
        Args:
            mllm_bot: MLLM bot for multimodal reasoning.
            target_image_path (str): Path to the target image.
            train_samples (dict): {category: [image_paths]} for references.
            superclass (str): Superclass (e.g., 'dog').
            kshot (int): Number of references.
            s (int): Number of regions.
            label (str, optional): Human-readable label name.
            label_id (int, optional): Label ID.
        
        Returns:
            str: Final attribute-aware description A_i.
        """
        # Step 1: Select references
        ref_paths = self.select_references(target_image_path, train_samples, kshot)
        
        # Step 2: Discover regions
        regions = self.discover_regions(mllm_bot, target_image_path, ref_paths, superclass, s)
        
        # Step 3: Describe attributes
        descriptions = self.describe_attributes(mllm_bot, target_image_path, regions, superclass, label, label_id)
        
        # Step 4: Summarize
        final_description = self.summarize_attributes(mllm_bot, target_image_path, descriptions, superclass, regions, label, label_id)
        
        return final_description

# Example usage (for testing)
if __name__ == "__main__":
    # Assume config or paths are set
    captioner = CDVCaptioner()
    mllm_bot = MLLMBot(model_tag="qwen", device='cuda')
    # Dummy train_samples dict
    train_samples = {"cat1": ["path/to/img1.jpg", "path/to/img2.jpg"], "cat2": ["path/to/img3.jpg"]}
    description = captioner.generate_description(mllm_bot, "path/to/target.jpg", train_samples, superclass="dog", t=5, s=3)
    print(description)