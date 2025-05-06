"""
模型加载器模块
负责优先加载本地模型，必要时才从网络下载
"""

import os
import torch
from diffusers import StableDiffusionPipeline, AutoencoderKL, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer
import logging

logger = logging.getLogger(__name__)

def find_local_model(model_name_or_path, search_dirs=None):
    """
    在指定的搜索目录中查找本地模型
    
    Args:
        model_name_or_path (str): 模型名称或路径
        search_dirs (list): 要搜索的目录列表，如果为None则使用默认目录
        
    Returns:
        str or None: 找到的本地模型路径，如果未找到则返回None
    """
    # 如果已经是本地路径并且存在，则直接返回
    if os.path.exists(model_name_or_path):
        if os.path.isdir(model_name_or_path):
            return model_name_or_path
    
    # 默认搜索目录
    if search_dirs is None:
        search_dirs = [
            os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub"),
            os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "transformers"),
            os.path.join(os.path.expanduser("~"), "models"),
            os.path.join("models"),
            os.path.join("pretrained_models"),
        ]
    
    # 如果model_name包含斜杠(如CompVis/stable-diffusion-v1-4)，提取模型名称部分
    model_name = model_name_or_path.split("/")[-1] if "/" in model_name_or_path else model_name_or_path
    
    # 在所有搜索目录中查找模型
    for search_dir in search_dirs:
        if not os.path.exists(search_dir):
            continue
            
        # 检查直接匹配的目录
        direct_path = os.path.join(search_dir, model_name)
        if os.path.exists(direct_path) and os.path.isdir(direct_path):
            # 检查是否包含模型文件
            if os.path.exists(os.path.join(direct_path, "model_index.json")) or \
               os.path.exists(os.path.join(direct_path, "config.json")):
                return direct_path
        
        # 检查带组织名称的目录 (如 CompVis--stable-diffusion-v1-4)
        if "/" in model_name_or_path:
            org_name = model_name_or_path.split("/")[0]
            org_model_path = os.path.join(search_dir, f"{org_name}--{model_name}")
            if os.path.exists(org_model_path) and os.path.isdir(org_model_path):
                # 检查是否包含模型文件
                if os.path.exists(os.path.join(org_model_path, "model_index.json")) or \
                   os.path.exists(os.path.join(org_model_path, "config.json")):
                    return org_model_path
        
        # 在搜索目录中遍历所有子目录，查找匹配模型名称的目录
        for root, dirs, _ in os.walk(search_dir):
            for d in dirs:
                if model_name.lower() in d.lower():  # 模糊匹配
                    path = os.path.join(root, d)
                    if os.path.exists(os.path.join(path, "model_index.json")) or \
                       os.path.exists(os.path.join(path, "config.json")):
                        return path
    
    # 未找到匹配的本地模型
    return None

def load_model_with_local_priority(model_name_or_path, model_type="pipeline", **kwargs):
    """
    优先加载本地模型，如果不存在则从网络下载
    
    Args:
        model_name_or_path (str): 模型名称或路径
        model_type (str): 加载的模型类型 ("pipeline", "vae", "unet", "text_encoder", "tokenizer")
        **kwargs: 传递给模型加载函数的其他参数
        
    Returns:
        模型对象
    """
    # 查找本地模型
    local_path = find_local_model(model_name_or_path)
    
    if local_path:
        print(f"找到本地模型: {local_path}")
        model_path = local_path
    else:
        print(f"未找到本地模型，将尝试从网络加载: {model_name_or_path}")
        model_path = model_name_or_path
    
    try:
        print(f"正在加载模型: {model_path}")
        
        # 根据模型类型加载不同的模型
        if model_type == "pipeline":
            model = StableDiffusionPipeline.from_pretrained(model_path, **kwargs)
            # 保存模型路径，方便后续使用
            model.model_path = model_path
        elif model_type == "vae":
            model = AutoencoderKL.from_pretrained(model_path, **kwargs)
        elif model_type == "unet":
            model = UNet2DConditionModel.from_pretrained(model_path, **kwargs)
        elif model_type == "text_encoder":
            model = CLIPTextModel.from_pretrained(model_path, **kwargs)
        elif model_type == "tokenizer":
            model = CLIPTokenizer.from_pretrained(model_path, **kwargs)
        else:
            raise ValueError(f"不支持的模型类型: {model_type}")
        
        print(f"模型加载成功: {model_path}")
        return model
        
    except Exception as e:
        print(f"加载模型时出错: {str(e)}")
        
        if local_path:
            print("本地模型加载失败，尝试从网络重新下载...")
            try:
                model = load_model_from_network(model_name_or_path, model_type, **kwargs)
                return model
            except Exception as e2:
                print(f"从网络加载模型也失败: {str(e2)}")
                raise e2
        else:
            raise e

def load_model_from_network(model_name_or_path, model_type="pipeline", **kwargs):
    """
    从网络加载模型
    
    Args:
        model_name_or_path (str): 模型名称或路径
        model_type (str): 加载的模型类型
        **kwargs: 传递给模型加载函数的其他参数
        
    Returns:
        模型对象
    """
    print(f"从网络加载模型: {model_name_or_path}")
    
    # 设置更长的超时时间
    kwargs['local_files_only'] = False
    
    if model_type == "pipeline":
        model = StableDiffusionPipeline.from_pretrained(model_name_or_path, **kwargs)
        model.model_path = model_name_or_path
    elif model_type == "vae":
        model = AutoencoderKL.from_pretrained(model_name_or_path, **kwargs)
    elif model_type == "unet":
        model = UNet2DConditionModel.from_pretrained(model_name_or_path, **kwargs)
    elif model_type == "text_encoder":
        model = CLIPTextModel.from_pretrained(model_name_or_path, **kwargs)
    elif model_type == "tokenizer":
        model = CLIPTokenizer.from_pretrained(model_name_or_path, **kwargs)
    else:
        raise ValueError(f"不支持的模型类型: {model_type}")
    
    return model

def load_stable_diffusion_components(model_name_or_path, torch_dtype=torch.float16, **kwargs):
    """
    加载Stable Diffusion模型的各个组件
    
    Args:
        model_name_or_path (str): 模型名称或路径
        torch_dtype: 模型权重的数据类型
        **kwargs: 其他参数
        
    Returns:
        tuple: (tokenizer, text_encoder, vae, unet)
    """
    local_path = find_local_model(model_name_or_path)
    if local_path:
        print(f"使用本地模型组件: {local_path}")
        model_path = local_path
    else:
        print(f"未找到本地模型组件，尝试从网络加载: {model_name_or_path}")
        model_path = model_name_or_path
    
    try:
        # 加载各个组件
        tokenizer = load_model_with_local_priority(
            model_path, model_type="tokenizer", subfolder="tokenizer", **kwargs
        )
        
        text_encoder = load_model_with_local_priority(
            model_path, model_type="text_encoder", subfolder="text_encoder", 
            torch_dtype=torch_dtype, **kwargs
        )
        
        vae = load_model_with_local_priority(
            model_path, model_type="vae", subfolder="vae", 
            torch_dtype=torch_dtype, **kwargs
        )
        
        unet = load_model_with_local_priority(
            model_path, model_type="unet", subfolder="unet", 
            torch_dtype=torch_dtype, **kwargs
        )
        
        return tokenizer, text_encoder, vae, unet
    
    except Exception as e:
        print(f"加载模型组件时出错: {str(e)}")
        raise
