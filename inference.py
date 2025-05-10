"""
DreamBooth 高质量推理模块
提供更高质量图像生成的推理功能
"""
import torch # Ensure torch is imported for device detection
import os # Ensure os is imported
import numpy as np # Ensure numpy is imported
from PIL import Image # Ensure PIL is imported
# Import from specific diffusers paths as recommended by Pylance
from diffusers.pipelines.auto_pipeline import AutoPipelineForText2Image
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from diffusers.schedulers.scheduling_dpmsolver_multistep import DPMSolverMultistepScheduler
from diffusers.schedulers.scheduling_euler_ancestral_discrete import EulerAncestralDiscreteScheduler
from diffusers.schedulers.scheduling_euler_discrete import EulerDiscreteScheduler
from diffusers.schedulers.scheduling_pndm import PNDMScheduler
from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler
import json # Added for loading config by default
import argparse # Keep argparse for --prompt

def get_device():
    """Checks for CUDA availability and returns the appropriate device string."""
    if torch.cuda.is_available():
        # Using "cuda" is generally preferred as it defaults to the current CUDA device.
        # PyTorch handles the specific device index (like cuda:0) internally in most cases.
        print("CUDA is available. Using device: cuda")
        return "cuda"
    else:
        print("CUDA not available. Using device: cpu")
        return "cpu"

def load_pipeline_from_config(config, device):
    """
    从配置和训练输出加载推理pipeline
    
    Args:
        config: 配置字典
        device: 推理设备 (e.g., "cuda:0" or "cpu")
        
    Returns:
        diffusers.DiffusionPipeline: 加载的模型pipeline
    """
    model_base_path = config.get("paths", {}).get("output_dir")
    if not model_base_path:
        raise ValueError("Configuration 'paths.output_dir' is missing.")
    
    output_dir_candidate = os.path.join(model_base_path, "diffusers_model")

    def is_valid_local_model_dir(p):
        if not os.path.exists(p) or not os.path.isdir(p):
            return False
        # Check for common diffusers model files
        has_model_index = os.path.exists(os.path.join(p, "model_index.json"))
        has_unet_config = os.path.exists(os.path.join(p, "unet", "config.json"))
        # Add other checks if necessary, e.g., for specific components
        return has_model_index or has_unet_config

    if not is_valid_local_model_dir(output_dir_candidate):
        print(f"未找到有效本地模型于 {output_dir_candidate}，尝试使用预训练模型路径。")
        output_dir_candidate = config.get("paths", {}).get("pretrained_model_name_or_path")
        if not output_dir_candidate:
            raise ValueError("Configuration 'paths.pretrained_model_name_or_path' is missing and primary model path is invalid.")
        
        # If it's not a valid local dir, assume it might be an HF ID or a path that from_pretrained can handle
        if not is_valid_local_model_dir(output_dir_candidate) and not os.path.isdir(output_dir_candidate):
            print(f"路径 {output_dir_candidate} 不是有效本地目录，假设为 Hugging Face 模型ID或可直接加载的路径。")
        elif not is_valid_local_model_dir(output_dir_candidate):
             raise FileNotFoundError(f"无法在主要路径或预训练路径 {output_dir_candidate} 找到有效本地模型。")

    output_dir = output_dir_candidate
    print(f"从模型路径加载: {output_dir}")
    
    torch_dtype = torch.float16
    if config.get("inference", {}).get("high_precision", False):
        torch_dtype = torch.float32
        print("使用全精度(fp32)进行推理以提高质量。")
    
    print(f"使用 AutoPipelineForText2Image 加载模型...")
    try:
        pipe = AutoPipelineForText2Image.from_pretrained(
            output_dir,
            torch_dtype=torch_dtype,
            safety_checker=None,
        )
    except TypeError as te: # Specifically catch TypeErrors that might relate to safety_checker
        print(f"使用 AutoPipelineForText2Image 加载模型时遇到 TypeError (可能与 safety_checker 有关): {te}")
        print("尝试不带 safety_checker=None 参数加载...")
        pipe = AutoPipelineForText2Image.from_pretrained(
            output_dir,
            torch_dtype=torch_dtype
        )
        if hasattr(pipe, 'safety_checker') and pipe.safety_checker is not None:
            print("警告: 模型加载了 safety_checker。")
    except Exception as e:
        print(f"使用 AutoPipelineForText2Image 加载模型失败: {e}")
        raise # Re-raise other exceptions

    scheduler_type = config.get("inference", {}).get("scheduler", "DPMSolverMultistep")
    print(f"尝试设置调度器为: {scheduler_type}")
    
    try:
        new_scheduler_instance = None        # When loading a scheduler from a model's pretrained path (HF ID or local model root),
        # the scheduler's config is typically in a "scheduler" subfolder.
        # The 'output_dir' variable here refers to the model's identifier (e.g., HF ID or local model root path).
        scheduler_load_kwargs = {"subfolder": "scheduler", "return_unused_kwargs": False}
        print(f"将尝试从模型路径 \\'{output_dir}\\' 的子目录 \\'{scheduler_load_kwargs['subfolder']}\\' 加载调度器配置。")

        if scheduler_type == "DDIM":
            new_scheduler_instance = DDIMScheduler.from_pretrained(output_dir, **scheduler_load_kwargs)
        elif scheduler_type == "DPMSolverMultistep":
            new_scheduler_instance = DPMSolverMultistepScheduler.from_pretrained(output_dir, **scheduler_load_kwargs)
            if config.get("inference", {}).get("use_karras_sigmas", True):
                # Check if the scheduler itself has the attribute or method
                if hasattr(new_scheduler_instance, 'config'):
                    # Check if it's a dict or an object with attributes
                    if isinstance(new_scheduler_instance.config, dict):
                        # It's a dict, so use dict syntax
                        new_scheduler_instance.config['use_karras_sigmas'] = True
                        print("已在调度器配置字典中启用 Karras sigmas。")
                    elif hasattr(new_scheduler_instance.config, 'use_karras_sigmas'):
                        # It's an object with the attribute
                        new_scheduler_instance.config.use_karras_sigmas = True
                        print("已在调度器配置对象中启用 Karras sigmas。")
                    else:
                        print(f"警告: 配置请求 use_karras_sigmas=True 但调度器 {scheduler_type} 的配置对象不支持此属性。")
                else:
                    print(f"警告: 配置请求 use_karras_sigmas=True 但调度器 {scheduler_type} 不支持此功能。")
        elif scheduler_type == "EulerAncestral":
            new_scheduler_instance = EulerAncestralDiscreteScheduler.from_pretrained(output_dir, **scheduler_load_kwargs)
        elif scheduler_type == "Euler":
            new_scheduler_instance = EulerDiscreteScheduler.from_pretrained(output_dir, **scheduler_load_kwargs)
        elif scheduler_type == "PNDM":
            new_scheduler_instance = PNDMScheduler.from_pretrained(output_dir, **scheduler_load_kwargs)
        elif scheduler_type == "UniPC":
            new_scheduler_instance = UniPCMultistepScheduler.from_pretrained(output_dir, **scheduler_load_kwargs)
        else:
            print(f"未知的调度器类型: {scheduler_type}。将使用模型默认加载的调度器: {pipe.scheduler.__class__.__name__}")
        
        if new_scheduler_instance:
            pipe.scheduler = new_scheduler_instance
            print(f"最终使用的调度器: {pipe.scheduler.__class__.__name__}")
        
    except Exception as e:
        print(f"加载/设置自定义调度器 {scheduler_type} 时出错: {e}。将使用模型默认加载的调度器: {pipe.scheduler.__class__.__name__}")

    print(f"模型将加载到设备: {device}")
    pipe = pipe.to(device)
    
    if config.get("inference", {}).get("attention_slicing", True):
        if hasattr(pipe, "enable_attention_slicing"):
            pipe.enable_attention_slicing()
            print("已启用注意力切片 (如果支持)。")
        else:
            print("注意: enable_attention_slicing 在当前 pipeline 类型中不可用。")
    
    if config.get("memory_optimization", {}).get("xformers_optimization", True):
        if hasattr(pipe, "enable_xformers_memory_efficient_attention"):
            try:
                pipe.enable_xformers_memory_efficient_attention()
                print("已启用 xFormers 内存优化 (如果支持且可用)。")
            except Exception as e:
                print(f"无法启用 xFormers: {e}。")
        else:
            print("注意: enable_xformers_memory_efficient_attention 在当前 pipeline 类型中不可用。")
            
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
    
    # 生成图像
    images = []
    for i in range(0, num_images, max(1, config["inference"].get("vae_batch_size", num_images))):
        batch_size = min(num_images - i, config["inference"].get("vae_batch_size", num_images))
        
        # 重新设计的高质量模式 - 更简单但更可靠
        if config["inference"].get("high_quality_mode", False):
            print(f"批次 {i+1}/{num_images}: 使用增强的高质量模式")
            
            # 高质量模式的增强参数
            hq_steps = int(num_inference_steps * 1.5)  # 50% 更多的步数
            hq_guidance = guidance_scale * 1.1  # 略微提高引导尺度
            
            # 增强负面提示
            enhanced_negative = negative_prompt
            if "deformed, disfigured, malformed" not in enhanced_negative:
                enhanced_negative = negative_prompt + ", deformed, disfigured, malformed limbs, missing limbs, bad anatomy, bad proportions"
            
            print(f"· 增强步数: {hq_steps}")
            print(f"· 调整引导尺度: {hq_guidance}")
            print(f"· 使用增强的负面提示")
            
            # 单阶段高质量生成
            try:
                batch_images = pipe(
                    prompt=prompt,
                    negative_prompt=enhanced_negative,
                    height=height,
                    width=width,
                    num_images_per_prompt=batch_size,
                    num_inference_steps=hq_steps,
                    guidance_scale=hq_guidance,
                    generator=generator,
                    noise_offset=noise_offset if noise_offset > 0 else 0.05,  # 小的噪声偏移有时可以帮助改善细节
                ).images
                
                images.extend(batch_images)
            except Exception as e:
                print(f"高质量生成出错: {e}, 回退到标准模式")
                batch_images = pipe(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    height=height,
                    width=width,
                    num_images_per_prompt=batch_size,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    generator=generator,
                ).images
                images.extend(batch_images)
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

def perform_inference(model_dir, prompt, num_images=1, seed=None, height=512, width=512, config=None, device=None):
    """
    执行完整的推理流程：加载模型、生成图像并保存
    
    Args:
        model_dir: 模型目录
        prompt: 提示词
        num_images: 要生成的图像数量
        seed: 随机种子
        height: 图像高度
        width: 图像宽度
        config: 配置对象，如果未提供将构建默认配置
        device: The device to run inference on (e.g., "cuda" or "cpu")
        
    Returns:
        list: 保存的图像路径列表
    """
    if device is None:
        device = get_device() # Get device if not provided
    print(f"Perform inference using device: {device}")

    # 如果没有提供配置，创建一个基本配置
    if config is None:
        try:
            # 尝试从model_dir相关的目录中加载config.json
            # Assuming model_dir is like 'dreambooth_output/run_id/diffusers_model'
            # config.json would be 'dreambooth_output/run_id/config.json' or just 'config.json'
            # For simplicity, we'll assume config is passed or loaded before this function
            # If this function is called directly, it might need a more robust way to find config
            # However, in the main script flow, config will be loaded first.
            
            # Default config if not passed and not easily found relative to model_dir
            print("Warning: perform_inference called without a config object. Creating a default one.")
            config = {
                "paths": {"output_dir": model_dir}, # model_dir here is the specific model path
                "inference": {
                    "num_inference_steps": 50,
                    "guidance_scale": 7.5,
                    "negative_prompt": "ugly, blurry, poor quality",
                    "scheduler": "DPMSolverMultistep",
                    "use_karras_sigmas": True,
                    "noise_offset": 0.1,
                    "vae_batch_size": 1,
                    "high_quality_mode": False
                },
                "memory_optimization": {"xformers_optimization": True}
            }
        except Exception as e:
            print(f"创建配置时出错: {e}")
            # 最小化的配置
            config = {
                "paths": {"output_dir": model_dir}, 
                "inference": {},
                "memory_optimization": {}
            }

    # 确保配置包含必要的路径
    # The output_dir in config usually refers to the base output directory for the training run
    # model_dir passed to this function is the specific subdirectory for the diffusers model
    # We need to ensure config["paths"]["output_dir"] is the base for finding generated_images
    # For saving, we use the model_dir directly.
    
    # 确定设备 - now passed as an argument or determined above
    # device = "cuda:0" if torch.cuda.is_available() else "cpu" # Old logic
    # print(f"使用设备: {device}") # Already printed
    
    try:
        # 加载模型
        print(f"正在加载模型...")
        pipe = load_pipeline_from_config(config, device) # Pass the determined device
        
        # 生成图像
        print(f"开始生成图像...")
        images = generate_high_quality_images(
            pipe,
            prompt,
            config,
            height=height,
            width=width,
            num_images=num_images,
            seed=seed
        )
        
        # 保存图像
        # The 'model_dir' argument to perform_inference is the root for 'generated_images'
        generated_images_output_dir = os.path.join(model_dir, "generated_images") 
        prefix = prompt.split()[:3]  # 使用提示词的前几个词作为前缀
        prefix = "_".join(prefix).replace(",", "").replace(".", "")
        saved_paths = save_images(images, generated_images_output_dir, prefix=prefix)
        
        return saved_paths
        
    except Exception as e:
        print(f"推理过程中出错: {e}")
        import traceback
        traceback.print_exc()
        return []

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="DreamBooth 高质量推理")
    # parser.add_argument("--config", type=str, default="config.json", help="配置文件路径") # Removed
    parser.add_argument("--prompt", type=str, help="生成提示 (覆盖config中的validation_prompt)")
    parser.add_argument("--num_images", type=int, help="要生成的图像数量 (覆盖config)")
    parser.add_argument("--seed", type=int, default=None, help="随机种子 (覆盖config)")
    parser.add_argument("--height", type=int, help="图像高度 (覆盖config)")
    parser.add_argument("--width", type=int, help="图像宽度 (覆盖config)")
    
    args = parser.parse_args()
    
    # Determine device at the start of the script execution
    current_device = get_device()

    config_file_path = "config.json" # Default config file
    if not os.path.exists(config_file_path):
        print(f"错误: 默认配置文件 \'{config_file_path}\' 未找到.")
        exit(1)
        
    # 加载配置
    with open(config_file_path, "r") as f:
        config = json.load(f)
    print(f"已从 '{config_file_path}' 加载配置。")

    # 确定设备以供脚本使用
    if torch.cuda.is_available():
        current_cuda_device_id = torch.cuda.current_device()
        device_to_use = f"cuda:{current_cuda_device_id}"
        print(f"CUDA 可用。脚本将使用设备: {device_to_use}")
    else:
        device_to_use = "cpu"
        print(f"CUDA 不可用。脚本将使用设备: {device_to_use}")

    # 从配置中获取推理设置，并允许命令行参数覆盖
    infer_settings = config.get("inference", {})
    
    # Determine prompt: command line > config > default
    prompt_to_use = args.prompt
    if not prompt_to_use:
        prompt_to_use = config.get("logging_saving", {}).get("validation_prompt")
    if not prompt_to_use: # Fallback if not in logging_saving
        prompt_to_use = infer_settings.get("default_prompt", "a photo of a sks dog") # Or some other default

    num_images_to_generate = args.num_images if args.num_images is not None else infer_settings.get("num_images_per_prompt", 1)
    seed_to_use = args.seed # Can be None
    height_to_use = args.height if args.height is not None else infer_settings.get("image_height", 512)
    width_to_use = args.width if args.width is not None else infer_settings.get("image_width", 512)

    # model_dir should come from the config's output path
    model_output_base_dir = config.get("paths", {}).get("output_dir")
    if not model_output_base_dir:
        print("错误: 配置中未找到 \'paths.output_dir\'。无法确定模型位置。")
        exit(1)
    
    print(f"将使用模型基础目录: {model_output_base_dir}")
    print(f"提示词: {prompt_to_use}")
    print(f"生成图像数量: {num_images_to_generate}")
    if seed_to_use is not None:
        print(f"随机种子: {seed_to_use}")
    print(f"图像尺寸: {width_to_use}x{height_to_use}")

    # 调用核心推理功能
    image_paths = perform_inference(
        model_dir=model_output_base_dir, 
        prompt=prompt_to_use,
        num_images=num_images_to_generate,
        seed=seed_to_use,
        height=height_to_use,
        width=width_to_use,
        config=config, 
        device=current_device # Pass the determined device
    )
    
    if image_paths:
        print(f"\\n推理完成！")
    else:
        print(f"\\n推理失败或未生成图像。请查看上方错误信息。")
