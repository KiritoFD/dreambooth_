"""
DreamBooth 高质量推理模块
提供更高质量图像生成的推理功能
"""
import os
import torch
from PIL import Image
import numpy as np
from diffusers import (
    StableDiffusionPipeline, 
    DDIMScheduler, 
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    PNDMScheduler,
    UniPCMultistepScheduler
)

def load_pipeline_from_config(config, device="cuda"):
    """
    从配置和训练输出加载推理pipeline
    
    Args:
        config: 配置字典
        device: 推理设备, 默认为"cuda"
        
    Returns:
        StableDiffusionPipeline: 加载的模型pipeline
    """
    output_dir = os.path.join(config["paths"]["output_dir"], "diffusers_model")
    
    # 如果没有训练好的模型，则从预训练模型加载
    if not os.path.exists(output_dir):
        print(f"未找到训练好的模型，将从预训练模型 {config['paths']['pretrained_model_name_or_path']} 加载")
        output_dir = config["paths"]["pretrained_model_name_or_path"]
    else:
        print(f"从训练输出目录加载模型: {output_dir}")
    
    # 选择调度器
    scheduler_type = config["inference"].get("scheduler", "DPMSolverMultistep")
    print(f"使用调度器: {scheduler_type}")
    
    if scheduler_type == "DDIM":
        scheduler = DDIMScheduler.from_pretrained(output_dir, subfolder="scheduler")
    elif scheduler_type == "DPMSolverMultistep":
        scheduler = DPMSolverMultistepScheduler.from_pretrained(output_dir, subfolder="scheduler")
        # 如果启用了Karras sigmas，应用此设置提高质量
        if config["inference"].get("use_karras_sigmas", True):
            scheduler.use_karras_sigmas = True
            print("已启用 Karras sigmas 以提高质量")
    elif scheduler_type == "EulerAncestral":
        scheduler = EulerAncestralDiscreteScheduler.from_pretrained(output_dir, subfolder="scheduler")
    elif scheduler_type == "Euler":
        scheduler = EulerDiscreteScheduler.from_pretrained(output_dir, subfolder="scheduler")
    elif scheduler_type == "PNDM":
        scheduler = PNDMScheduler.from_pretrained(output_dir, subfolder="scheduler")
    elif scheduler_type == "UniPC":
        scheduler = UniPCMultistepScheduler.from_pretrained(output_dir, subfolder="scheduler")
    else:
        print(f"未知的调度器类型: {scheduler_type}，使用默认的 DPMSolverMultistep")
        scheduler = DPMSolverMultistepScheduler.from_pretrained(output_dir, subfolder="scheduler")
    
    # 确定是使用fp16还是fp32
    torch_dtype = torch.float16
    if config["inference"].get("high_precision", False):
        torch_dtype = torch.float32
        print("使用全精度(fp32)进行推理以提高质量")
    
    # 加载pipeline
    pipe = StableDiffusionPipeline.from_pretrained(
        output_dir,
        scheduler=scheduler,
        torch_dtype=torch_dtype,
        safety_checker=None  # 对于艺术创作可以禁用安全检查器
    )
    
    pipe = pipe.to(device)
    
    # 应用内存优化
    if config["inference"].get("attention_slicing", True):
        pipe.enable_attention_slicing()
        print("已启用注意力切片以节省内存")
    
    # 如果配置了xFormers，则使用之
    if config["memory_optimization"].get("xformers_optimization", True):
        try:
            pipe.enable_xformers_memory_efficient_attention()
            print("已启用 xFormers 内存优化")
        except:
            print("无法启用 xFormers，继续使用标准注意力")
    
    return pipe

def generate_high_quality_images(
    pipe, 
    prompt, 
    config,
    height=512,
    width=512,
    num_images=1,
    seed=None
):
    """
    生成高质量图像
    
    Args:
        pipe: StableDiffusionPipeline 实例
        prompt: 生成提示
        config: 配置字典
        height: 图像高度
        width: 图像宽度
        num_images: 要生成的图像数量
        seed: 随机种子，用于可复现生成
        
    Returns:
        list: 生成的图像列表
    """
    # 设置随机种子以实现可复现性
    if seed is not None:
        generator = torch.Generator(device=pipe.device).manual_seed(seed)
    else:
        generator = None
    
    # 从配置中获取推理参数
    num_inference_steps = config["inference"].get("num_inference_steps", 50)
    guidance_scale = config["inference"].get("guidance_scale", 7.5)
    negative_prompt = config["inference"].get("negative_prompt", "ugly, blurry, poor quality")
    noise_offset = config["inference"].get("noise_offset", 0.0)
    
    print(f"正在生成 {num_images} 张图像...")
    print(f"提示: {prompt}")
    print(f"否定提示: {negative_prompt}")
    print(f"步数: {num_inference_steps}, 引导尺度: {guidance_scale}")
    
    # 禁用进度条以避免日志污染
    pipe.set_progress_bar_config(disable=True)
    
    # 应用额外技巧来提高质量
    if "vae_batch_size" in config["inference"]:
        vae_batch_size = config["inference"]["vae_batch_size"]
        if vae_batch_size > 0 and vae_batch_size < num_images:
            print(f"使用VAE批处理大小: {vae_batch_size}")
            # 启用VAE切片以减少内存使用并提高稳定性
            pipe.enable_vae_slicing()
    
    if config["inference"].get("high_quality_mode", False):
        print("已启用高质量模式，生成可能较慢...")
    
    # 生成图像
    images = []
    for i in range(0, num_images, max(1, config["inference"].get("vae_batch_size", num_images))):
        batch_size = min(num_images - i, config["inference"].get("vae_batch_size", num_images))
        
        # 根据是否使用高质量模式选择不同的生成策略
        if config["inference"].get("high_quality_mode", False):
            # 第一阶段: 轮廓生成
            print(f"批次 {i+1}/{num_images}: 第1阶段 - 生成基础轮廓...")
            first_pass = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                height=height,
                width=width,
                num_images_per_prompt=batch_size,
                num_inference_steps=int(num_inference_steps * 0.7),  # 使用较少步数
                guidance_scale=guidance_scale * 0.8,  # 较低的引导尺度以获得更多样性
                generator=generator
            ).images
            
            # 第二阶段: 细节增强
            print(f"批次 {i+1}/{num_images}: 第2阶段 - 增强细节...")
            latents = pipe.vae.encode([pipe.image_processor.preprocess(img) for img in first_pass]).latent_dist.sample()
            latents = latents * 0.18215  # 缩放因子
            
            batch_images = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                height=height,
                width=width,
                num_images_per_prompt=batch_size,
                num_inference_steps=num_inference_steps,  # 使用完整步数
                guidance_scale=guidance_scale,
                latents=latents,  # 使用第一阶段的隐变量作为起点
                generator=generator,
                noise_offset=noise_offset if noise_offset > 0 else None,
            ).images
        else:
            # 普通模式: 直接生成
            print(f"批次 {i+1}/{num_images}: 标准生成模式")
            batch_images = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                height=height,
                width=width,
                num_images_per_prompt=batch_size,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=generator,
                noise_offset=noise_offset if noise_offset > 0 else None,
            ).images
        
        images.extend(batch_images)
    
    print(f"成功生成 {len(images)} 张图像")
    return images

def save_images(images, output_dir, prefix="generated"):
    """保存生成的图像到指定目录"""
    os.makedirs(output_dir, exist_ok=True)
    
    saved_paths = []
    for i, img in enumerate(images):
        timestamp = np.datetime64('now').astype(str).replace(':', '-').replace(' ', '_')
        filename = f"{prefix}_{timestamp}_{i+1}.png"
        path = os.path.join(output_dir, filename)
        img.save(path)
        saved_paths.append(path)
        print(f"已保存图像到 {path}")
    
    return saved_paths

if __name__ == "__main__":
    import argparse
    import json
    
    parser = argparse.ArgumentParser(description="DreamBooth 高质量推理")
    parser.add_argument("--config", type=str, default="config.json", help="配置文件路径")
    parser.add_argument("--prompt", type=str, required=True, help="生成提示")
    parser.add_argument("--num_images", type=int, default=1, help="要生成的图像数量")
    parser.add_argument("--seed", type=int, default=None, help="随机种子")
    parser.add_argument("--height", type=int, default=512, help="图像高度")
    parser.add_argument("--width", type=int, default=512, help="图像宽度")
    
    args = parser.parse_args()
    
    # 加载配置
    with open(args.config, "r") as f:
        config = json.load(f)
    
    # 如果配置中没有inference部分，添加默认值
    if "inference" not in config:
        config["inference"] = {
            "num_inference_steps": 50,
            "guidance_scale": 7.5,
            "negative_prompt": "ugly, blurry, poor quality",
            "scheduler": "DPMSolverMultistep",
            "use_karras_sigmas": True,
            "noise_offset": 0.1,
            "vae_batch_size": 1,
            "high_quality_mode": True
        }
    
    # 确定设备
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")
    
    # 加载模型
    pipe = load_pipeline_from_config(config, device)
    
    # 生成图像
    images = generate_high_quality_images(
        pipe, 
        args.prompt, 
        config,
        height=args.height,
        width=args.width,
        num_images=args.num_images,
        seed=args.seed
    )
    
    # 保存图像
    output_dir = os.path.join(config["paths"]["output_dir"], "generated_images")
    save_images(images, output_dir)
    
    print(f"推理完成! 图像已保存到 {output_dir}")
