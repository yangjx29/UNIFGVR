
## UniFGVR: 细粒度视觉分类的统一多模态检索框架

UniFGVR 是一个用于细粒度视觉分类（FGVC）的统一多模态检索框架。它由两大模块组成：
- CDV-Captioner：参考引导 + 链式思考（CoT）的区分性视觉描述生成。
- Multimodal Retrieval：图像与文本融合特征，构建类别模板库并进行相似度检索。

> 基于FineR修改

### 特性
- 参考引导描述：在同一超类下，选取相似但不同类别的参考图像，引导 MLLM 生成更具区分性的属性描述。
- 多模态融合：结合图像编码器与文本编码器（如 CLIP）获取鲁棒的融合表示。
- 模板检索：以类别模板为中心进行相似度匹配，轻量高效，易于扩展新类别。
- 全流程可复现：提供清晰的配置、日志与输出目录结构，便于复现实验。

### 目录结构
```text
UniFGVR/
├─ agents/                  # 主要关注MLLM
├─ cvd/                     # CDV-Captioner 实现
├─ data/                    # 数据集适配与统计
├─ experiments/             # gallery等构造的文件在这
├─ retrieval/               # 多模态检索实现
├─ utils/                   
├─ configs/                 # 环境与实验配置
├─ logs/                    # 运行日志
└─ discovering.py           # main
```

## 环境依赖
- Python 3.10
- PyTorch
- Transformers, Pillow, NumPy, scikit-learn, tqdm, termcolor

安装示例：
参考 FineR
<!-- ```bash
# 按需选择你的 CUDA/CPU 版本
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

pip install transformers pillow numpy scikit-learn tqdm termcolor
``` -->

## 数据与模型准备
- 数据集：以 Dogs120 为例
```text
datasets/dogs_120/Images/
  ├─ n02085620-Chihuahua/
  ├─ n02085782-Japanese_spaniel/
  └─ ...
```

- 编码器与模型
  - 图像/文本编码器：默认使用 CLIP（在 `retrieval/multimodal_retrieval.py` 中修改模型路径/名称）。
  - MLLM：在 `agents/mllm_bot.py` 配置（例如 `Qwen2.5-VL-8B`），并设置运行设备。
  - 配置文件：
    - 环境配置：`configs/env_machine.yml`
    - 实验配置：`configs/expts/dog120_all.yml`

## 快速开始

### 1) 构建类别模板库（Gallery）
将 K-shot 样本通过 CDV-Captioner 转换为图像-描述对，提取图像/文本特征，融合后按类别平均得到模板。

```bash
CUDA_VISIBLE_DEVICES=0 \
python discovering.py \
  --mode=build_gallery \
  --config_file_env=./configs/env_machine.yml \
  --config_file_expt=./configs/expts/dog120_all.yml \
  --kshot=5 --region_num=3 --superclass=dog \
  --gallery_out=./experiments/dog120/gallery/dog120_gallery.json \
  --fusion_method=concat
  2>&1 | tee ./logs/build_gallery_dog.log
```

输出格式：
```json
{
  "Chihuahua": [0.0123, -0.0456, ...],
  "Shiba Inu": [0.0234, -0.0345, ...]
}
路径：./experiments/dog120/gallery/dog120_gallery.json：
```

### 2) 检索与评估

- 加载已构建的 gallery 并进行 FGVC 检索/评估：
```bash
python retrieval/multimodal_retrieval.py 2>&1 | tee ./logs/inference_dog.log
```

- 测试RAG功能（Top-5候选 + MLLM推理）：
```bash
python test_rag.py
```

#### RAG检索流程
1. **多模态相似度计算**：计算查询图像与所有类别模板的余弦相似度
2. **Top-5候选选择**：选择相似度最高的5个类别作为候选
3. **MLLM推理**：构造RAG prompt，让MLLM基于候选类别和图像进行最终推理
4. **结果提取**：从MLLM输出中提取最终预测类别

相比简单的Top-1方法，RAG方法能够：
- 利用MLLM的视觉理解能力进行更准确的判断
- 在相似类别间进行更细致的区分
- 提供更好的可解释性（可以看到Top-5候选和最终推理过程）
