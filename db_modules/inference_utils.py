"""
DreamBooth 高质量推理工具
提供使用保留的最佳模型进行高质量图像生成的功能
"""
import os
import torch
import numpy as np
from PIL import Image
from diffusers import (
    StableDiffusionPipeline, 
    DDIMScheduler, 
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    PNDMScheduler,
    UniPCMultistepScheduler
)

def find_best_model_checkpoint(output_dir):
    """查找输出目录中的最佳模型"""
    # 首先检查是否有final模型
    final_model_dir = os.path.join(output_dir, "diffusers_model")
    if os.path.exists(final_model_dir):
        return final_model_dir, "最终保存的模型"
    
    # 检查是否有checkpoint目录
    checkpoint_dir = os.path.join(output_dir, "checkpoints")
    if not os.path.exists(checkpoint_dir):
        return None, "未找到模型检查点"
    
    # 查找最新的checkpoint
    checkpoints = [d for d in os.listdir(checkpoint_dir) if os.path.isdir(os.path.join(checkpoint_dir, d))]
    if not checkpoints:
        return None, "检查点目录为空"
    
    # 按照创建时间排序
    checkpoints.sort(key=lambda x: os.path.getctime(os.path.join(checkpoint_dir, x)), reverse=True)
    latest_checkpoint = os.path.join(checkpoint_dir, checkpoints[0])
    return latest_checkpoint, f"最新检查点 ({checkpoints[0]})"

def load_inference_pipeline(config, device="cuda"):
    """
    加载推理pipeline，优先使用保存的最佳模型
    
    Args:
        config: 配置字典
        device: 推理设备, 默认为"cuda"
        
    Returns:
        tuple: (pipeline, model_path, model_info)
    """
    output_dir = config["paths"]["output_dir"]
    
    # 查找最佳模型
    model_path, model_info = find_best_model_checkpoint(output_dir)
    
    # 如果没有找到模型，使用预训练模型
    if model_path is None:
        print(f"未找到训练好的模型，将从预训练模型 {config['paths']['pretrained_model_name_or_path']} 加载")
        model_path = config["paths"]["pretrained_model_name_or_path"]
        model_info = "预训练模型"
    else:
        print(f"加载模型: {model_path}")
        print(f"模型信息: {model_info}")
    
    # 选择调度器
    scheduler_type = config["inference"].get("scheduler", "DPMSolverMultistep")
    print(f"使用调度器: {scheduler_type}")
    
    # 创建调度器
    if scheduler_type == "DDIM":
        scheduler = DDIMScheduler.from_pretrained(model_path, subfolder="scheduler")
    elif scheduler_type == "DPMSolverMultistep":
        scheduler = DPMSolverMultistepScheduler.from_pretrained(model_path, subfolder="scheduler")
        # 配置Karras sigmas以提高质量
        if config["inference"].get("use_karras_sigmas", True):
            scheduler.use_karras_sigmas = True
            print("已启用 Karras sigmas 以提高质量")
    elif scheduler_type == "EulerAncestral":
        scheduler = EulerAncestralDiscreteScheduler.from_pretrained(model_path, subfolder="scheduler")
    elif scheduler_type == "Euler":
        scheduler = EulerDiscreteScheduler.from_pretrained(model_path, subfolder="scheduler")
    elif scheduler_type == "PNDM":
        scheduler = PNDMScheduler.from_pretrained(model_path, subfolder="scheduler")
    elif scheduler_type == "UniPC":
        scheduler = UniPCMultistepScheduler.from_pretrained(model_path, subfolder="scheduler")
    else:
        print(f"未知的调度器类型: {scheduler_type}，使用默认的 DPMSolverMultistep")
        scheduler = DPMSolverMultistepScheduler.from_pretrained(model_path, subfolder="scheduler")
    
    # 确定使用的精度
    torch_dtype = torch.float16
    if config["inference"].get("high_precision", False):
        torch_dtype = torch.float32
        print("使用全精度(fp32)进行推理以提高质量")
    
    # 加载pipeline
    try:
        pipe = StableDiffusionPipeline.from_pretrained(
            model_path,
            scheduler=scheduler,
            torch_dtype=torch_dtype,
            safety_checker=None
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
        
        return pipe, model_path, model_info
    
    except Exception as e:
        print(f"加载模型发生错误: {e}")
        print("尝试不使用自定义调度器加载模型...")
        
        pipe = StableDiffusionPipeline.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            safety_checker=None
        )
        pipe = pipe.to(device)
        return pipe, model_path, f"{model_info} (使用默认调度器)"

def generate_images(
    pipe, 
    prompt, 
    config,
    negative_prompt=None,
    height=512,
    width=512,
    num_images=1,
    seed=None,
    output_dir=None
):
    """
    生成高质量图像
    
    Args:
        pipe: StableDiffusionPipeline 实例
        prompt: 生成提示
        config: 配置字典
        negative_prompt: 否定提示，默认使用配置中的值
        height: 图像高度
        width: 图像宽度
        num_images: 要生成的图像数量
        seed: 随机种子，用于可复现生成
        output_dir: 输出目录
        
    Returns:
        list: 生成的图像路径列表
    """
    # 设置随机种子以实现可复现性
    if seed is not None:
        generator = torch.Generator(device=pipe.device).manual_seed(seed)
        print(f"使用随机种子: {seed}")
    else:
        generator = None
        print("使用随机种子")
    
    # 从配置中获取推理参数
    num_inference_steps = config["inference"].get("num_inference_steps", 50)
    guidance_scale = config["inference"].get("guidance_scale", 7.5)
    
    # 如果未提供否定提示，使用配置中的值
    if negative_prompt is None:
        negative_prompt = config["inference"].get("negative_prompt", "ugly, blurry, poor quality")
    
    noise_offset = config["inference"].get("noise_offset", 0.1)
    
    # 显示生成参数
    print(f"正在生成 {num_images} 张图像...")
    print(f"提示: {prompt}")
    print(f"否定提示: {negative_prompt}")
    print(f"步数: {num_inference_steps}, 引导尺度: {guidance_scale}")
    print(f"图像尺寸: {width}x{height}")
    
    # 禁用进度条以避免日志污染
    pipe.set_progress_bar_config(disable=False)
    
    # 应用VAE批处理以提高性能
    vae_batch_size = config["inference"].get("vae_batch_size", num_images)
    if vae_batch_size > 0 and vae_batch_size < num_images:
        print(f"使用VAE批处理大小: {vae_batch_size}")
        pipe.enable_vae_slicing()
    
    high_quality_mode = config["inference"].get("high_quality_mode", True)
    if high_quality_mode:
        print("已启用高质量模式，生成可能较慢...")
    
    # 生成图像
    all_images = []
    for i in range(0, num_images, max(1, vae_batch_size)):
        batch_size = min(num_images - i, vae_batch_size)
        
        # 根据是否使用高质量模式选择不同的生成策略
        if high_quality_mode:
            # 第一阶段：轮廓生成
            print(f"批次 {i+1}/{num_images}: 阶段1 - 生成基础轮廓...")
            first_pass = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                height=height,
                width=width,
                num_images_per_prompt=batch_size,
                num_inference_steps=int(num_inference_steps * 0.6),  # 60%的步数
                guidance_scale=guidance_scale * 0.8,  # 较低引导尺度
                generator=generator
            ).images
            
            # 第二阶段：细节增强
            print(f"批次 {i+1}/{num_images}: 阶段2 - 增强细节...")
            
            # 将第一阶段的图像转换为隐空间表示
            first_pass_tensors = [pipe.image_processor.preprocess(img) for img in first_pass]
            
            # 检查设备兼容性
            first_pass_tensors = [tensor.to(pipe.device) for tensor in first_pass_tensors]
            
            # 编码到隐空间
            latents = []
            for tensor in first_pass_tensors:
                with torch.no_grad():
                    latent = pipe.vae.encode(tensor.unsqueeze(0)).latent_dist.sample()
                    latents.append(latent)
            
            latents = torch.cat(latents, dim=0)
            latents = latents * 0.18215  # 缩放因子
            
            # 使用第一阶段的隐变量作为起点，进行第二阶段生成
            batch_images = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                height=height,
                width=width,
                num_images_per_prompt=batch_size,
                num_inference_steps=num_inference_steps,  # 完整步数
                guidance_scale=guidance_scale,
                latents=latents,
                generator=generator,
                noise_offset=noise_offset if noise_offset > 0 else None
            ).images
        else:
            # 普通模式：直接生成
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
                noise_offset=noise_offset if noise_offset > 0 else None
            ).images
        
        all_images.extend(batch_images)
    
    # 保存图像
    if output_dir is None:
        output_dir = os.path.join(config["paths"]["output_dir"], "generated_images")
    
    os.makedirs(output_dir, exist_ok=True)
    
    saved_paths = []
    timestamp = np.datetime64('now').astype(str).replace(':', '-').replace(' ', '_')
    
    for i, img in enumerate(all_images):
        filename = f"generated_{timestamp}_{i+1}.png"
        path = os.path.join(output_dir, filename)
        img.save(path)
        saved_paths.append(path)
        print(f"已保存图像到: {path}")
    
    print(f"成功生成并保存 {len(saved_paths)} 张图像到 {output_dir}")
    return saved_paths
