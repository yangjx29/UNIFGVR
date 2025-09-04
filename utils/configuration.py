import os  
import torch  
import torch.cuda 
import yaml  # 导入YAML配置文件解析模块
from easydict import EasyDict  # 导入EasyDict，提供字典的点号访问功能
import numpy as np  # 导入NumPy数值计算库
import random  # 导入随机数生成模块
from utils.fileios import mkdir_if_missing  # 导入创建目录的工具函数


def setup_config(config_file_env: str, config_file_expt: str):
    """设置配置函数：合并环境配置和实验配置"""
    with open(config_file_env, 'r') as stream:  # 打开环境配置文件
        config_env = yaml.safe_load(stream) 

    with open(config_file_expt, 'r') as stream: 
        config_expt = yaml.safe_load(stream) 

    cfg_env = EasyDict()  
    cfg_expt = EasyDict()  

    # Copy - 复制配置
    for k, v in config_env.items():  # 遍历环境配置项
        cfg_env[k] = v  # 复制到环境配置字典

    for k, v in config_expt.items():  # 遍历实验配置项
        cfg_expt[k] = v  # 复制到实验配置字典

    # Init. configuration - 初始化配置
    #   |- device - 设备配置
    cfg_expt['host'] = cfg_env['host']  # 设置主机名
    cfg_expt['num_workers'] = cfg_env['num_workers']  # 设置工作进程数

    if torch.cuda.is_available():  # 如果CUDA可用
        cfg_expt['device'] = "cuda"  # 设置设备为CUDA
        cfg_expt['device_count'] = f"{torch.cuda.device_count()}"  # 设置GPU数量
        cfg_expt['device_id'] = f"{torch.cuda.current_device()}"  # 设置当前GPU设备ID
    else:  # 否则使用CPU
        cfg_expt['device'] = "cpu"  # 设置设备为CPU

    #   |- file paths - 文件路径配置
    if cfg_expt['dataset_name'] == "bird":  # 如果是鸟类数据集
        cfg_expt['data_dir'] = os.path.join(cfg_env['data_root'], "CUB_200_2011/CUB_200_2011/")  # 设置数据目录
        cfg_expt['expt_dir'] = os.path.join(cfg_env['expt_root'], "bird200")  # 设置实验目录
        mkdir_if_missing(cfg_expt['expt_dir'])  # 创建实验目录（如果不存在）
    elif cfg_expt['dataset_name'] == "dog":  # 如果是狗数据集
        cfg_expt['data_dir'] = os.path.join(cfg_env['data_root'], "dogs_120")  # 设置数据目录
        cfg_expt['expt_dir'] = os.path.join(cfg_env['expt_root'], "dog120")  # 设置实验目录
        mkdir_if_missing(cfg_expt['expt_dir'])  # 创建实验目录（如果不存在）
    elif cfg_expt['dataset_name'] == "pet":  # 如果是宠物数据集
        cfg_expt['data_dir'] = os.path.join(cfg_env['data_root'], "pet_37")  # 设置数据目录
        cfg_expt['expt_dir'] = os.path.join(cfg_env['expt_root'], "pet37")  # 设置实验目录
        mkdir_if_missing(cfg_expt['expt_dir'])  # 创建实验目录（如果不存在）
    elif cfg_expt['dataset_name'] == "flower":  # 如果是花数据集
        cfg_expt['data_dir'] = os.path.join(cfg_env['data_root'], "flowers_102")  # 设置数据目录
        cfg_expt['expt_dir'] = os.path.join(cfg_env['expt_root'], "flower102")  # 设置实验目录
        mkdir_if_missing(cfg_expt['expt_dir'])  # 创建实验目录（如果不存在）
    elif cfg_expt['dataset_name'] == "car":  # 如果是汽车数据集
        cfg_expt['data_dir'] = os.path.join(cfg_env['data_root'], "car_196")  # 设置数据目录
        cfg_expt['expt_dir'] = os.path.join(cfg_env['expt_root'], "car196")  # 设置实验目录
        mkdir_if_missing(cfg_expt['expt_dir'])  # 创建实验目录（如果不存在）
    elif cfg_expt['dataset_name'] == "food":  # 如果是食物数据集
        cfg_expt['data_dir'] = os.path.join(cfg_env['data_root'], "food_101")  # 设置数据目录
        cfg_expt['expt_dir'] = os.path.join(cfg_env['expt_root'], "food101")  # 设置实验目录
        mkdir_if_missing(cfg_expt['expt_dir'])  # 创建实验目录（如果不存在）
    elif cfg_expt['dataset_name'] == "place":  # 如果是场景数据集
        cfg_expt['data_dir'] = os.path.join(cfg_env['data_root'], "place_365")  # 设置数据目录
        cfg_expt['expt_dir'] = os.path.join(cfg_env['expt_root'], "place365")  # 设置实验目录
        mkdir_if_missing(cfg_expt['expt_dir'])  # 创建实验目录（如果不存在）
    elif cfg_expt['dataset_name'] == "pokemon":  # 如果是宝可梦数据集
        cfg_expt['data_dir'] = os.path.join(cfg_env['data_root'], "pokemon")  # 设置数据目录
        cfg_expt['expt_dir'] = os.path.join(cfg_env['expt_root'], "pokemon")  # 设置实验目录
        mkdir_if_missing(cfg_expt['expt_dir'])  # 创建实验目录（如果不存在）
    else:  # 如果是未知数据集
        raise NameError(f"{cfg_expt['dataset_name']} is a wrong dataset name")

    # for Stage Discovery - 为发现阶段配置
    cfg_expt['expt_dir_describe'] = os.path.join(cfg_expt['expt_dir'], "describe")  # 设置描述阶段实验目录
    mkdir_if_missing(cfg_expt['expt_dir_describe'])  # 创建描述阶段目录（如果不存在）
    cfg_expt['path_vqa_questions'] = os.path.join(cfg_expt['expt_dir_describe'],  # 设置VQA问题保存路径
                                                  f"{cfg_expt['dataset_name']}_vqa_questions_ours")
    cfg_expt['path_vqa_answers'] = os.path.join(cfg_expt['expt_dir_describe'],  # 设置VQA答案保存路径
                                                f"{cfg_expt['dataset_name']}_attributes_pairs")
    cfg_expt['path_llm_prompts'] = os.path.join(cfg_expt['expt_dir_describe'],  # 设置LLM提示保存路径
                                                f"{cfg_expt['dataset_name']}_llm_prompts")
                                        
    cfg_expt['path_identify_answers'] = os.path.join(cfg_expt['expt_dir'], "identify")

    #for Stage Guess - 为猜测阶段配置
    cfg_expt['expt_dir_guess'] = os.path.join(cfg_expt['expt_dir'], "guess")  # 设置猜测阶段实验目录
    mkdir_if_missing(cfg_expt['expt_dir_guess'])  # 创建猜测阶段目录（如果不存在）
    cfg_expt['path_llm_replies_raw'] = os.path.join(cfg_expt['expt_dir_guess'],  # 设置LLM原始回复保存路径
                                                    f"{cfg_expt['dataset_name']}_llm_replies_raw")
    cfg_expt['path_llm_replies_jsoned'] = os.path.join(cfg_expt['expt_dir_guess'],  # 设置LLM JSON回复保存路径
                                                       f"{cfg_expt['dataset_name']}_llm_replies_jsoned")
    cfg_expt['path_llm_gussed_names'] = os.path.join(cfg_expt['expt_dir_guess'],  # 设置LLM猜测名称保存路径
                                                     f"{cfg_expt['dataset_name']}_llm_gussed_names")
    cfg_expt['expt_dir_gallery'] = os.path.join(cfg_expt['expt_dir'], "gallery")
    cfg_expt['path_references'] = os.path.join(cfg_expt['expt_dir_gallery'], 
                                            f"{cfg_expt['dataset_name']}_image_references")
    cfg_expt['path_regions'] = os.path.join(cfg_expt['expt_dir_gallery'], 
                                            f"{cfg_expt['dataset_name']}_image_regions")
    cfg_expt['path_descriptions'] = os.path.join(cfg_expt['expt_dir_gallery'], 
                                    f"{cfg_expt['dataset_name']}_image_descriptions")


    # for Stage Grouping evaluation - 为分组评估阶段配置
    cfg_expt['expt_dir_grouping'] = os.path.join(cfg_expt['expt_dir'], "grouping")  # 设置分组阶段实验目录
    mkdir_if_missing(cfg_expt['expt_dir_grouping'])  # 创建分组阶段目录（如果不存在）

    #   |- model - 模型配置
    if cfg_expt['model_size'] == 'ViT-L/14@336px' and cfg_expt['image_size'] != 336:  # 如果是ViT-L/14@336px模型但图像尺寸不是336
        print(
            f'Model size is {cfg_expt["model_size"]} but image size is {cfg_expt["image_size"]}. Setting image size to 336.')  # 打印警告信息
        cfg_expt['image_size'] = 336  # 设置图像尺寸为336
    elif cfg_expt['model_size'] == 'RN50x4' and cfg_expt['image_size'] != 288:  # 如果是RN50x4模型但图像尺寸不是288
        print(
            f'Model size is {cfg_expt["model_size"]} but image size is {cfg_expt["image_size"]}. Setting image size to 288.')  # 打印警告信息
        cfg_expt['image_size'] = 288  # 设置图像尺寸为288
    elif cfg_expt['model_size'] == 'RN50x16' and cfg_expt['image_size'] != 384:  # 如果是RN50x16模型但图像尺寸不是384
        print(
            f'Model size is {cfg_expt["model_size"]} but image size is {cfg_expt["image_size"]}. Setting image size to 288.')  # 打印警告信息
        cfg_expt['image_size'] = 384  # 设置图像尺寸为384
    elif cfg_expt['model_size'] == 'RN50x64' and cfg_expt['image_size'] != 448:  # 如果是RN50x64模型但图像尺寸不是448
        print(
            f'Model size is {cfg_expt["model_size"]} but image size is {cfg_expt["image_size"]}. Setting image size to 288.')  # 打印警告信息
        cfg_expt['image_size'] = 448  # 设置图像尺寸为448

    #   |- data augmentation - 数据增强配置
    return cfg_expt  # 返回实验配置


def seed_everything(seed: int):
    """设置所有随机种子以确保实验可重现性"""
    random.seed(seed)  # 设置Python随机数种子
    os.environ['PYTHONHASHSEED'] = str(seed)  # 设置Python哈希种子环境变量
    np.random.seed(seed)  # 设置NumPy随机数种子
    torch.manual_seed(seed)  # 设置PyTorch CPU随机数种子
    torch.cuda.manual_seed(seed)  # 设置PyTorch GPU随机数种子
    torch.backends.cudnn.deterministic = True  # 设置CUDNN为确定性模式
    torch.backends.cudnn.benchmark = True  # 启用CUDNN基准测试