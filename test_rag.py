#!/usr/bin/env python3
"""
测试RAG功能的简单脚本
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from retrieval.multimodal_retrieval import MultimodalRetrieval
from cvd.cdv_captioner import CDVCaptioner
from agents.mllm_bot import MLLMBot
import torch

def test_rag_functionality():
    """测试RAG功能"""
    print("=== 测试RAG功能 ===")
    
    # 初始化组件
    retrieval = MultimodalRetrieval(fusion_method="concat")
    captioner = CDVCaptioner()
    
    # 根据环境选择设备
    device = "cuda" if torch.cuda.is_available() else "cpu"
    mllm_bot = MLLMBot(
        model_tag="Qwen2.5-VL-8B", 
        model_name="Qwen2.5-VL-8B", 
        device=device
    )
    
    # 加载已构建的gallery
    gallery_path = "./experiments/dog120/gallery/dog120_gallery.json"
    if not os.path.exists(gallery_path):
        print(f"错误：找不到gallery文件 {gallery_path}")
        print("请先运行 discovering.py --mode=build_gallery 构建gallery")
        return
    
    gallery = retrieval.load_gallery_from_json(gallery_path)
    print(f"加载了 {len(gallery)} 个类别的gallery")
    
    # 测试单张图像
    test_image = "/data/yjx/MLLM/UniFGVR/datasets/dogs_120/Images/n02085620-Chihuahua/n02085620_1558.jpg"
    
    if not os.path.exists(test_image):
        print(f"错误：找不到测试图像 {test_image}")
        return
    
    print(f"\n测试图像: {test_image}")
    
    # 测试Top-1方法
    print("\n=== Top-1 方法 ===")
    pred_top1, scores_top1 = retrieval.fgvc_via_multimodal_retrieval(
        mllm_bot, test_image, gallery, captioner, "dog", use_rag=False
    )
    
    # 测试RAG方法
    print("\n=== RAG 方法 (Top-5 + MLLM推理) ===")
    pred_rag, scores_rag = retrieval.fgvc_via_multimodal_retrieval(
        mllm_bot, test_image, gallery, captioner, "dog", use_rag=True
    )
    
    print(f"\n结果对比:")
    print(f"Top-1 预测: {pred_top1}")
    print(f"RAG 预测: {pred_rag}")
    
    # 显示Top-5候选
    top5_candidates = sorted(scores_rag.items(), key=lambda x: x[1], reverse=True)[:5]
    print(f"\nTop-5 候选类别:")
    for i, (cat, score) in enumerate(top5_candidates, 1):
        print(f"  {i}. {cat}: {score:.4f}")

if __name__ == "__main__":
    test_rag_functionality()
