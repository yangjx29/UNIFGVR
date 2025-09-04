import torch 
import os 
import argparse 
import json 
from tqdm import tqdm  
from termcolor import colored  
from collections import Counter 
from utils.configuration import setup_config, seed_everything 
from utils.fileios import dump_json, load_json, dump_txt  

from data import DATA_STATS, PROMPTERS, DATA_DISCOVERY  
from data.prompt_identify import prompts_howto  
from agents.vqa_bot import VQABot  
from agents.llm_bot import LLMBot 
from agents.mllm_bot import MLLMBot
from cvd.cdv_captioner import CDVCaptioner  
from retrieval.multimodal_retrieval import MultimodalRetrieval 
import re 
import hashlib
from collections import defaultdict
import numpy as np


DEBUG = False  # 设置调试模式为关闭状态


def cint2cname(label: int, cname_sheet: list):
    """将类别整数索引转换为类别名称"""
    return cname_sheet[label]


def extract_superidentify(cfg, individual_results):
    """从个体识别结果中提取超类识别结果"""
    words = []  # 初始化单词列表
    for v in individual_results.values():  # 遍历所有个体识别结果
        this_word = v.split(' ')[-1]  # 取最后一个单词作为类别标识
        words.append(this_word.lower())  # 转换为小写并添加到列表
    word_counts = Counter(words)  # 统计每个单词的出现次数
    # print(f"extract_superidentify 中每个单词出现次数: {word_counts}")
    if cfg['dataset_name'] == 'pet':  # 如果是宠物数据集
        return [super_name for super_name, _ in word_counts.most_common(2)]  # 返回出现次数最多的2个超类
    else:  # 其他数据集
        return [super_name for super_name, _ in word_counts.most_common(1)]  # 返回出现次数最多的1个超类



def extract_python_list(text):
    """从文本中提取Python列表格式的内容"""
    pattern = r"\[(.*?)\]"  # 定义匹配方括号内容的正则表达式
    matches = re.findall(pattern, text)  # 查找所有匹配的内容
    return matches  # 返回匹配结果列表


def trim_result2json(raw_reply: str):
    """
    the raw_answer is a dirty output from LLM following our template.
    this function helps to extract the target JSON content contained in the
    output.
    """
    # 从LLM的原始输出中提取JSON格式的内容
    if raw_reply.find("Output JSON:") >= 0:  # 如果包含"Output JSON:"标记
        answer = raw_reply.split("Output JSON:")[1].strip()  # 提取标记后的内容
    else:  # 否则直接使用原始内容
        answer = raw_reply.strip()  # 去除首尾空白字符

    if not answer.startswith('{'): answer = '{' + answer  # 如果开头不是{，则添加

    if not answer.endswith('}'): answer = answer + '}'  # 如果结尾不是}，则添加

    # json_answer = json.loads(answer)  # 注释掉的JSON解析代码
    return answer  # 返回处理后的JSON字符串


def clean_name(name: str):
    """清理类别名称，统一格式"""
    name = name.title() 
    name = name.replace("-", " ")  
    name = name.replace("'s", "") 
    return name  


def extract_names(gussed_names, clean=True):
    """从猜测的名称列表中提取和清理名称"""
    gussed_names = [name.strip() for name in gussed_names]
    if clean:  # 如果需要清理
        gussed_names = [clean_name(name) for name in gussed_names]  
    gussed_names = list(set(gussed_names))  # 去重并转换为列表
    return gussed_names  # 返回处理后的名称列表


def how_to_distinguish(bot, prompt):
    """询问LLM如何区分不同类别"""
    reply = bot.infer(prompt, temperature=0.1) 
    used_tokens = bot.get_used_tokens()  
    print(f"llm used_tokens: {used_tokens},")
    print(20*"=")  
    print(reply)  #
    print(20*"=") 

    return reply  

def main_identify(cfg, bot, data_disco):
    """识别图像的超类"""
    json_super_classes = {}             # img: [attr1, attr2, ..., attrN] - 初始化超类结果字典

    # print(f"现在开始遍历发现集data_disco: {data_disco}")
    for idx, (img, label) in tqdm(enumerate(data_disco)):  # 遍历发现数据集中的图像和标签
        # prompt_identify = "Question: What is the main object in this image (choose from: Car, Flower, or Pokemon)? Answer:"
        
        prompt_identify = "Question: What is the category (car, bird, flower, dog, cat, or Pokemon) of the main object in this image? Answer:" 

        reply, trimmed_reply = bot.describe_attribute(img, prompt_identify) 
        trimmed_reply = trimmed_reply.lower()  
        json_super_classes[str(idx)] = trimmed_reply 

        # DEBUG mode - 调试模式
        if DEBUG and idx >= 2: 
            break  
    # print(f"main_identify 识别结果: {json_super_classes}")
    return json_super_classes  # 返回超类识别结果


def main_describe(cfg, bot, data_disco, prompter, cname_sheet):
    """
    1.调用VQA模型为每个属性生成对应的描述
    2.生成LLMpromot描述
    """
    json_attrs = {}             # img: [attr1, attr2, ..., attrN] - 初始化属性结果字典
    json_llm_prompts = {}       # img: LLM-prompt (has all attrs) - 初始化LLM提示字典

    # 这里是训练集，预先定义好的
    for idx, (img, label) in tqdm(enumerate(data_disco)): 
        if cfg['dataset_name'] == 'pet': 
            # first check what is the animal
            pet_prompt = "Questions: What is the animal in this photo (dog or car)? Answer:"
            pet_re, pet_trimmed_re = bot.describe_attribute(img, pet_prompt) 
            pet_trimmed_re = pet_trimmed_re.lower() 
            # print(pet_trimmed_re)
            if 'dog' in pet_trimmed_re:
                prompter.set_superclass('dog')
            else:
                prompter.set_superclass('cat')

        # generate attributes and per-attribute prompts for VQA bot  获得属性列表
        attrs = prompter.get_attributes()
        # 生成对应属性的promot描述，让LLM进行描述
        attr_prompts = prompter.get_attribute_prompt() 
        if len(attrs) != len(attr_prompts):  # 检查属性列表和提示列表长度是否一致
            raise IndexError("Attribute list should have the same length as attribute prompts")

        print(f"当前idx:{idx}: label={label}")

        iname = cint2cname(label, cname_sheet)
        iname += f"_{idx}"  # 对应用多少个样本作为训练集
        json_attrs[iname] = []  # 初始化该图像的属性列表

        # describe each attrs - 描述每个属性
        pair_attr_reply = []    # (attr1: prompt) - 初始化属性-值对列表
        for attr, p_attr in zip(attrs, attr_prompts):  # 遍历属性和对应的prompt
            re_attr, trimmed_re_attr = bot.describe_attribute(img, p_attr) 
            # print(f"调用bot.describe_attribute得到的reply:{re_attr} \n tritrimmed_re_attr:{trimmed_re_attr}")
            pair_attr_reply.append([attr, trimmed_re_attr])
            json_attrs[iname].append(trimmed_re_attr)  # 将属性值添加到对应的类别描述中
        
        print(f'获得的VQA pair_attr_reply: {pair_attr_reply}\n json_attrs: {json_attrs}')
        # generate LLM prompt - 生成LLM提示
        llm_prompt = prompter.get_llm_prompt(pair_attr_reply)  # 根据属性-值对生成LLM提示
        json_llm_prompts[iname] = llm_prompt 
        print(f'json_llm_prompts: {json_llm_prompts}')
        print(30 * '=')
        print(iname + f" with label {label}") 
        print(30 * '=')
        # print()  # 打印空行
        # print(f"llm_prompt: {llm_prompt}")  # 打印LLM提示
        # print()  # 打印空行
        # print('END' + 30 * '=')  # 打印结束分隔线
        # print()  # 打印空行

        # DEBUG mode - 调试模式
        if DEBUG and idx >= 2:  # 如果开启调试模式且处理了2个以上样本
            break  # 跳出循环

    return json_attrs, json_llm_prompts  # 返回属性结果和LLM提示


def main_guess(cfg, bot, reasoning_prompts):
    """主要猜测函数：基于属性描述推理类别名称"""
    prompt_list = reasoning_prompts  
    replies_raw = {}  
    replies_json_to_save = {}  

    # LLM inferring - LLM推理
    for i, (key, prompt) in tqdm(enumerate(prompt_list.items())):  
        raw_reply = bot.infer(prompt, temperature=0.9)  # use a high temperature for better diversity
        used_tokens = bot.get_used_tokens()  # 获取使用的token数量

        replies_raw[key] = raw_reply  # 将原始回复存储到字典

        print(30 * '=')  # 打印分隔线
        print(f"\t\tinferring [{i}] for {key} used tokens = {used_tokens}") 
        print(30 * '=')  
        print("Raw----")  
        print(raw_reply)  
        print()  

        jsoned_reply = trim_result2json(raw_reply=raw_reply) 

        replies_json_to_save[key] = jsoned_reply  

        print("Trimed----")  
        print(jsoned_reply)
        print()  # 打印空行
        print('END' + 30 * '=')  
        print()  

        # DEBUG - 调试
        if DEBUG and i >= 2:  # 如果开启调试模式且处理了2个以上样本
            break 

    print(30 * '=')  
    print(f"\t\t Finish Discovering, token consumed {llm_bot.get_used_tokens()}"  
          f" = ${bot.get_used_tokens()*0.001*0.002}") 
    print(30 * '=')  
    print('END' + 30 * '=')  
    print()  
    return replies_raw, replies_json_to_save 


def post_process(cfg, jsoned_replies):
    """后处理函数：清理和整理LLM推理结果"""
    reply_list = []  
    num_of_failures = 0  
    # duplicated dict - 重复字典
    for k, v in jsoned_replies.items():  # 遍历JSON回复字典
        print(k)  
        print(v) 
        print()
        print() 
        try:  
            v_json = json.loads(v)  
            reply_list.append(v_json)
        except json.JSONDecodeError:  
            print(f"Failed to decode JSON for key: {k}") 
            num_of_failures += 1  
            continue  

        # v_json = json.loads(v) 
        # reply_list.append(v_json) 

    guessed_names = [] 
    for item in reply_list: 
        guessed_names.extend(list(item.keys()))  

    guessed_names = extract_names(guessed_names, clean=False) 

    if cfg['dataset_name'] in ['pet', 'dog']: 
        clean_gussed_names = []  
        for aitem in guessed_names:
            clean_gussed_names.extend(aitem.split(','))  
        clean_gussed_names = [name.strip() for name in clean_gussed_names]  
        guessed_names = clean_gussed_names  

    print(30 * '=') 
    print(f"\t\t Finished Post-processing")  
    print(30 * '=')  

    print(f"\t\t ---> total discovered names = {len(guessed_names)}")  
    print(guessed_names)  
    print()  
    print(f"\t\t ---> total discovered names = {len(guessed_names)}")  
    print(f"\t\t ---> number of failure entries = {num_of_failures}") 

    print('END' + 30 * '=')  
    print()  
    return guessed_names 


def load_train_samples(cfg, kshot=None):
    """加载K-shot训练样本，返回 {category: [image_paths]}。
    优先从 cfg['path_train_samples'] (JSON) 读取；否则从 cfg['train_root'] 目录扫描。
    """
    samples = {}
    if 'path_train_samples' in cfg and os.path.exists(cfg['path_train_samples']):
        try:
            samples = load_json(cfg['path_train_samples'])
        except Exception as e:
            print(f"failed to load path_train_samples: {cfg['path_train_samples']}, err={e}")
            samples = {}
    elif 'train_root' in cfg and os.path.isdir(cfg['train_root']):
        train_root = cfg['train_root']
        valid_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
        for cname in sorted(os.listdir(train_root)):
            cdir = os.path.join(train_root, cname)
            if not os.path.isdir(cdir):
                continue
            imgs = []
            for fname in sorted(os.listdir(cdir)):
                fpath = os.path.join(cdir, fname)
                ext = os.path.splitext(fname)[1].lower()
                if os.path.isfile(fpath) and ext in valid_exts:
                    imgs.append(fpath)
            if imgs:
                samples[cname] = imgs
    else:
        raise FileNotFoundError("Neither cfg['path_train_samples'] nor cfg['train_root'] is valid.")

    if kshot is not None:
        trimmed = {}
        for cat, paths in samples.items():
            trimmed[cat] = paths[:kshot]
        return trimmed
    return samples


def build_gallery(cfg, mllm_bot, captioner, retrieval, kshot=5,region_num=3, superclass=None, data_discovery=None):
    """构建多模态类别模板库并保存到JSON(向量转list)。"""

    # 读取训练样本
    k = kshot if kshot is not None else int(str(cfg.get('k_shot', '3')))
    # train_samples = load_train_samples(cfg, kshot=k)
    train_samples = defaultdict(list)
    for name, path in data_discovery.subcat_to_sample.items():
        train_samples[name].append(path)
    print(f"loaded train samples for {len(train_samples)} classes, kshot={k}")
    print(f"train_samples: {train_samples}") 

    # 构建模板库
    gallery = retrieval.build_template_gallery(mllm_bot, train_samples, captioner, superclass, kshot, region_num)
    
    return gallery

if __name__ == "__main__":
    """
    CUDA_VISIBLE_DEVICES=2 python discovering.py --mode=build_gallery --config_file_env=./configs/env_machine.yml --config_file_expt=./configs/expts/dog120_all.yml --kshot=5 --region_num=3 --superclass=dog  --gallery_out=./experiments/dog120/gallery/dog120_gallery.json 2>&1 | tee ./logs/build_gallery_dog.log
    """
    parser = argparse.ArgumentParser(description='Discovery', formatter_class=argparse.ArgumentDefaultsHelpFormatter) 

    parser.add_argument('--mode',  
                        type=str, 
                        default='describe', 
                        choices=['identify', 'howto', 'describe', 'guess', 'postprocess', 'build_gallery'],  # 可选值列表
                        help='operating mode for each stage')  
    parser.add_argument('--config_file_env',  
                        type=str,  
                        default='./configs/env_machine.yml',  # 默认配置文件路径
                        help='location of host environment related config file')  
    parser.add_argument('--config_file_expt',  # 添加实验配置文件参数
                        type=str,  
                        default='./configs/expts/bird200_all.yml', 
                        help='location of host experiment related config file') 
    # arguments for control experiments - 控制实验的参数
    parser.add_argument('--num_per_category',  # 添加每个类别的样本数量参数
                        type=str, 
                        default='3',  
                        choices=['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'random'], 
                        )
    # build_gallery 相关
    parser.add_argument('--kshot', type=int, default=None, help='shots per class when building gallery (override cfg)')
    parser.add_argument('--region_num', type=int, default=None, help='region selelct per class when building gallery (override cfg)')
    parser.add_argument('--superclass', type=str, default=None, help='superclass for CDV prompts (override cfg)')
    parser.add_argument('--gallery_out', type=str, default=None, help='path to save built gallery json')


    args = parser.parse_args()  
    print(colored(args, 'blue'))  

    cfg = setup_config(args.config_file_env, args.config_file_expt)  
    print(colored(cfg, 'yellow')) 

    # drop the seed - 设置随机种子
    seed_everything(cfg['seed']) 

    expt_id_suffix = f"_{args.num_per_category}"  # 创建实验ID后缀

    if args.mode == 'identify':  # 如果模式是识别
        # build VQA Bot - 构建VQA机器人
        if cfg['host'] in ["chaos", "YOUR_GPU_CLUSTER_NAME"]:
            vqa_bot = VQABot(model_tag=cfg['model_size_vqa'], device='cuda', device_id=cfg['device_id'], bit8=False) 
        else: 
            vqa_bot = VQABot(model_tag=cfg['model_size_vqa'], device='cpu') 
        cname_sheet = DATA_STATS[cfg['dataset_name']]['class_names']
        data_discovery = DATA_DISCOVERY[cfg['dataset_name']](cfg, folder_suffix=expt_id_suffix)  # 创建发现数据集
        save_path_identify_answers = cfg['path_identify_answers'] + expt_id_suffix  # 构建识别答案保存路径
        print(f"save_path_identify_answers: {save_path_identify_answers}")
        # run the main program to describe the per-img attributes - 运行主程序描述每张图像的属性
        superclass_results = main_identify(cfg, vqa_bot, data_discovery)

        # 选择data_discovery出现最多的类
        identified_super_class = extract_superidentify(cfg, superclass_results)  

        print(f"识别出来的identified_super_class: {identified_super_class}")  # 打印识别的超类
        dump_json(save_path_identify_answers, {'superclass': identified_super_class})  # 保存识别结果到JSON文件
        print(f"Succ. dumped identified super-class values to {save_path_identify_answers}") 
    elif args.mode == 'howto':  # 识别属性
        save_path_vqa_questions = cfg['path_vqa_questions']  # 获取VQA问题保存路径
        print(f"save_path_vqa_questions: {save_path_vqa_questions}")
        llm_bot = LLMBot(model=cfg['model_type_llm'], temperature=0.5)  # llm类型，使用低温度参数
        
        # 加载上一步识别出来的超类
        superclass = load_json(cfg['path_identify_answers'] + expt_id_suffix)['superclass']  # 加载识别的超类
        print(f"第一步识别的超类: {superclass}")
        if len(superclass) > 1:  # 如果有多个超类（如宠物数据集）
            prompt = [  # 创建提示列表
                prompts_howto["pet"].replace('[__SUPERCLASS__]', superclass[0]),  # 替换第一个超类
                prompts_howto["pet"].replace('[__SUPERCLASS__]', superclass[1])   # 替换第二个超类
            ]
        else:  # 如果只有一个超类
            if 'bird' in superclass[0]:  # 如果是鸟类
                prompt = [prompts_howto['bird'].replace('[__SUPERCLASS__]', 'bird')]  # 使用鸟类提示
            elif 'car' in superclass[0]:  # 如果是汽车
                prompt = [prompts_howto['car'].replace('[__SUPERCLASS__]', 'car')]  # 使用汽车提示
            elif 'dog' in superclass[0]:  # 如果是狗
                prompt = [prompts_howto['dog'].replace('[__SUPERCLASS__]', 'dog')]  # 使用狗提示
            elif 'flower' in superclass[0]:  # 如果是花
                prompt = [prompts_howto['flower'].replace('[__SUPERCLASS__]', 'flower')]  # 使用花提示
            elif 'pokemon' in superclass[0]:  # 如果是宝可梦
                prompt = [prompts_howto['pokemon'].replace('[__SUPERCLASS__]', 'pokemon')]  # 使用宝可梦提示

        pattern = r'\[([^\]]*)\]'  # 定义匹配方括号内容的正则表达式
        for i, ppt in enumerate(prompt):  # 遍历提示列表
            vqa_questions = how_to_distinguish(llm_bot, prompt=ppt)  # 询问LLM如何区分
            matches = re.findall(pattern, vqa_questions)  # 查找匹配的内容
            print(f'matches之后的reply: {matches}')
            result = matches[0].strip().replace('\n', '').replace('"', "'").replace("', '", "','")  # 处理匹配结果

            if cfg['dataset_name'] == 'pet':  # 如果是宠物数据集
                dump_txt(save_path_vqa_questions.replace('pet_vqa_questions',  
                                                         f'pet_{superclass[i]}_vqa_questions.txt'), f'[{result}]')  
            else:  
                print(f'vqa question保存到: {save_path_vqa_questions}')
                dump_txt(save_path_vqa_questions, f'[{result}]')  
    elif args.mode == 'describe':  # 描述属性
        """
        describe the attributes - 描述属性
        """
        # build VQA Bot - 构建VQA机器人
        if cfg['host'] in ["chaos", "YOUR_GPU_CLUSTER_NAME"]: 
            vqa_bot = VQABot(model_tag=cfg['model_size_vqa'], device='cuda', device_id=cfg['device_id'], bit8=False)  # 创建GPU版本的VQA机器人
        else:  
            vqa_bot = VQABot(model_tag=cfg['model_size_vqa'], device='cpu')

        # get data ordered class name lookup - 获取对应超类的所有子类名称
        cname_sheet = DATA_STATS[cfg['dataset_name']]['class_names']

        # build data set - 构建训练集
        data_discovery = DATA_DISCOVERY[cfg['dataset_name']](cfg, folder_suffix=expt_id_suffix)  # 创建发现数据集

        # build VQAbot prompter - 构建Prompter类，这个类里面也有对应超类的属性列表？？
        prompter = PROMPTERS[cfg['dataset_name']](cfg)  # 创建对应数据集的提示器

        # paths to save per-image VQAbot answers (about attributes) and per-image LLM prompts - 保存每张图像的VQAmodel回答和LLM提示的路径
        save_path_vqa_answers = cfg['path_vqa_answers'] + expt_id_suffix  # 构建VQA答案保存路径
        save_path_llm_prompts = cfg['path_llm_prompts'] + expt_id_suffix  # 构建LLM提示保存路径
        print(f"save_path_vqa_answers:{save_path_vqa_answers}\nsave_path_llm_prompts:{save_path_llm_prompts}")
        # run the main program to describe the per-img attributes - 运行主程序描述每张图像的属性
        json_vqa_answers, json_llm_prompts = main_describe(cfg, vqa_bot, data_discovery, prompter, cname_sheet)  # 调用主要描述函数

        dump_json(save_path_vqa_answers, json_vqa_answers)  # 保存VQA答案到JSON文件
        print(f"Succ. dumped attribute values to {save_path_vqa_answers}")  # 打印成功保存VQA答案信息

        dump_json(save_path_llm_prompts, json_llm_prompts)  # 保存LLM提示到JSON文件
        print(f"Succ. dumped LLM prompts  to {save_path_llm_prompts}") 
    elif args.mode == 'guess':  # 根据属性描述推测类别
        """
        reason category names based on the attribute-description pairs - 基于属性描述对推理类别名称
        """
        reasoning_prompts = load_json(cfg['path_llm_prompts'] + expt_id_suffix) 
        llm_bot = LLMBot(model=cfg['model_type_llm'])

        # run the main program - 运行主程序
        raw_replies, jsoned_replies = main_guess(cfg, llm_bot, reasoning_prompts)

        # save LLM replis - 保存LLM回复
        dump_json(cfg['path_llm_replies_raw'] + expt_id_suffix, raw_replies) 
        dump_json(cfg['path_llm_replies_jsoned'] + expt_id_suffix, jsoned_replies) 
    elif args.mode == 'postprocess': 
        """
        clean the results a bit - 清理结果
        """
        jsoned_replies = load_json(cfg['path_llm_replies_jsoned'] + expt_id_suffix)  # 加载JSON格式回复
        # post-process data - 后处理数据
        gussed_names = post_process(cfg, jsoned_replies)  # 调用后处理函数
        # save LLM gussed names - 保存LLM猜测的名称
        dump_json(cfg['path_llm_gussed_names'] + expt_id_suffix, gussed_names)  # 保存猜测的名称
    elif args.mode == 'build_gallery':
        """
        构建多模态类别模板库(gallery)
        训练样本来源优先级:cfg['path_train_samples']
        输出默认 cfg['path_gallery']，可用 --gallery_out 覆盖
        """
        if cfg['host'] in ["xiao"]:
            mllm_bot = MLLMBot(model_tag=cfg['model_size_mllm'], model_name=cfg['model_size_mllm'], device='cuda', device_id=cfg['device_id'], bit8=False) 
        else: 
            mllm_bot = MLLMBot(model_tag=cfg['model_size_mllm'], model_name=cfg['model_size_mllm'], device='cpu')

        captioner = CDVCaptioner(cfg=cfg)
        retrieval = MultimodalRetrieval()
        data_discovery = DATA_DISCOVERY[cfg['dataset_name']](cfg, folder_suffix=expt_id_suffix)
        if args.superclass is None:
            superclass_results = main_identify(cfg, mllm_bot, data_discovery)  # 调用主要识别函数
            identified_super_class = extract_superidentify(cfg, superclass_results)
            args.superclass = identified_super_class
        gallery = build_gallery(cfg, mllm_bot, captioner, retrieval, kshot=args.kshot, region_num=args.region_num, superclass=args.superclass, data_discovery=data_discovery)
        print(f"build_gallery: {gallery}")
        save_path = args.gallery_out
        gallery_to_save_path = save_path if save_path is not None else cfg.get('path_gallery', './gallery.json')

        # 保存构建的检索库
        gallery_to_save = {}
        for cat, feat in gallery.items():
            # 统一转为 np.float32 array
            if isinstance(feat, torch.Tensor):
                arr = feat.detach().cpu().numpy()
            elif isinstance(feat, np.ndarray):
                arr = feat
            else:
                arr = np.asarray(feat, dtype=np.float32)
            arr32 = arr.astype(np.float32, copy=False)
            gallery_to_save[cat] = arr32.tolist()
        dump_json(gallery_to_save_path, gallery_to_save)
        print(f"Succ. dumped gallery to {gallery_to_save_path}")
    else:
        raise NotImplementedError 

