"""
模型工具函数模块
负责模型加载、下载和本地缓存管理
"""
import os
import shutil
import torch
from diffusers import StableDiffusionPipeline, AutoencoderKL, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer
from huggingface_hub import HfFolder, snapshot_download
from huggingface_hub.utils import LocalEntryNotFoundError

def ensure_local_model(model_id, local_dir=None, use_auth_token=None, revision=None):
    """
    确保模型在本地可用，如果不可用则尝试下载
    
    参数:
        model_id (str): 模型ID，如 'CompVis/stable-diffusion-v1-4'
        local_dir (str, optional): 本地存储目录，默认为None时将使用 ~/.cache/huggingface/
        use_auth_token: HF认证令牌
        revision (str, optional): 模型版本，默认为None使用main
        
    返回:
        str: 本地模型路径
    """
    # 默认使用huggingface缓存目录
    if local_dir is None:
        cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
    else:
        cache_dir = local_dir
    
    # 检查用户指定的本地目录
    if local_dir is not None and os.path.exists(local_dir):
        if os.path.exists(os.path.join(local_dir, "model_index.json")):
            print(f"使用本地模型: {local_dir}")
            return local_dir
    
    # 检查是否已缓存
    try:
        # 尝试获取已缓存的模型路径
        model_path = snapshot_download(
            repo_id=model_id,
            revision=revision,
            cache_dir=cache_dir,
            use_auth_token=use_auth_token,
            local_files_only=True  # 只使用本地文件
        )
        print(f"使用缓存模型: {model_path}")
        return model_path
    except LocalEntryNotFoundError:
        print(f"本地未找到模型 {model_id}，尝试下载...")
        
    # 尝试下载模型
    try:
        model_path = snapshot_download(
            repo_id=model_id,
            revision=revision,
            cache_dir=cache_dir,
            use_auth_token=use_auth_token
        )
        print(f"模型下载成功: {model_path}")
        return model_path
    except Exception as e:
        print(f"下载模型时出错: {str(e)}")
        print("尝试使用备用下载方式或检查网络连接")
        
        # 这里可以添加备用下载逻辑，如使用git clone等
        
        raise ValueError(
            f"无法下载或找到模型 {model_id}。请手动下载模型并放置在 {local_dir or '~/.cache/huggingface/hub'} 目录，"
            f"或者提供正确的本地模型路径。"
        )

def load_model_components(
    pretrained_model_name_or_path,
    local_model_path=None,
    use_auth_token=None,
    revision=None,
    torch_dtype=torch.float16
):
    """
    加载Stable Diffusion模型组件
    
    返回:
        tuple: (tokenizer, text_encoder, vae, unet)
    """
    model_path = pretrained_model_name_or_path
    
    # 如果提供了本地路径或需要确保本地存在
    if local_model_path is not None or not os.path.exists(pretrained_model_name_or_path):
        try:
            model_path = ensure_local_model(
                pretrained_model_name_or_path,
                local_model_path,
                use_auth_token,
                revision
            )
        except Exception as e:
            print(f"警告: 无法获取模型 {pretrained_model_name_or_path}: {str(e)}")
            # 如果指定了本地路径但模型不完整，则回退到原始路径
            if local_model_path and os.path.exists(local_model_path):
                model_path = local_model_path
                print(f"尝试使用指定的本地模型路径: {model_path}")
    
    print(f"从路径加载模型: {model_path}")
    
    # 加载模型组件
    try:
        tokenizer = CLIPTokenizer.from_pretrained(
            model_path, subfolder="tokenizer", use_auth_token=use_auth_token, revision=revision
        )
        
        text_encoder = CLIPTextModel.from_pretrained(
            model_path, subfolder="text_encoder", use_auth_token=use_auth_token, 
            revision=revision, torch_dtype=torch_dtype
        )
        
        vae = AutoencoderKL.from_pretrained(
            model_path, subfolder="vae", use_auth_token=use_auth_token,
            revision=revision, torch_dtype=torch_dtype
        )
        
        unet = UNet2DConditionModel.from_pretrained(
            model_path, subfolder="unet", use_auth_token=use_auth_token,
            revision=revision, torch_dtype=torch_dtype
        )
        
        return tokenizer, text_encoder, vae, unet
    except Exception as e:
        print(f"加载模型组件时出错: {str(e)}")
        raise
