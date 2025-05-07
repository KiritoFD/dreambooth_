"""
DreamBooth 推理模块
用于加载训练好的模型并生成新图像
"""
import os
import time
import torch
from diffusers import StableDiffusionPipeline, DDIMScheduler
from PIL import Image
import datetime

def perform_inference(
    model_dir, 
    prompt, 
    output_dir=None,
    num_images=1,
    height=512,
    width=512,
    num_inference_steps=50,
    guidance_scale=7.5,
    seed=None
):
    """
    执行推理以生成图像
    
    Args:
        model_dir: 模型保存的目录路径
        prompt: 推理提示词
        output_dir: 输出目录，如未指定则使用model_dir
        num_images: 生成图像的数量
        height: 生成图像高度
        width: 生成图像宽度
        num_inference_steps: 推理步数
        guidance_scale: 提示词引导强度
        seed: 随机种子，用于可重复结果
    
    Returns:
        生成的图像文件路径列表
    """
    # 检查模型目录是否存在
    if not os.path.exists(model_dir):
        raise FileNotFoundError(f"模型目录 '{model_dir}' 不存在，无法执行推理。")
    
    # 设置输出目录
    if output_dir is None:
        output_dir = os.path.join(model_dir, "inference_results")
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"正在从 '{model_dir}' 加载模型...")
    
    # 设置推理设备
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")
    
    try:
        # 加载模型和调度器
        scheduler = DDIMScheduler.from_pretrained(
            model_dir, 
            subfolder="scheduler",
            torch_dtype=torch.float16 if device == "cuda" else torch.float32
        )
        
        # 使用float16以节省显存，在CPU上保持float32
        dtype = torch.float16 if device == "cuda" else torch.float32
        
        # 加载完整的StableDiffusion pipeline
        pipeline = StableDiffusionPipeline.from_pretrained(
            model_dir,
            scheduler=scheduler,
            torch_dtype=dtype,
            safety_checker=None
        ).to(device)
        
        # 优化推理速度
        if device == "cuda":
            pipeline.enable_xformers_memory_efficient_attention()
        
        print(f"模型加载完成，开始执行推理...")
        print(f"提示词: \"{prompt}\"")
        
        # 设置随机种子以便结果可复现
        generator = None
        if seed is not None:
            generator = torch.Generator(device=device).manual_seed(seed)
            print(f"使用随机种子: {seed}")
        
        # 生成图像
        image_paths = []
        for i in range(num_images):
            # 生成推理开始时间戳
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # 生成图像
            output = pipeline(
                prompt=prompt,
                height=height,
                width=width,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=generator,
                num_images_per_prompt=1
            )
            
            # 处理并保存图像
            if output.images:
                for j, image in enumerate(output.images):
                    # 构建输出文件名
                    clean_prompt = "".join([c if c.isalnum() else "_" for c in prompt[:20]])
                    image_filename = f"{timestamp}_{clean_prompt}_{i+1}of{num_images}.png"
                    image_path = os.path.join(output_dir, image_filename)
                    
                    # 保存图像
                    image.save(image_path)
                    image_paths.append(image_path)
                    print(f"图像已保存: {image_path}")
            else:
                print(f"警告: 第 {i+1} 批次未能生成图像。")
        
        return image_paths
        
    except Exception as e:
        print(f"推理过程中出错: {str(e)}")
        import traceback
        traceback.print_exc()
        return []

if __name__ == "__main__":
    # 可直接运行此脚本进行测试
    import argparse
    parser = argparse.ArgumentParser(description="DreamBooth 模型推理")
    parser.add_argument("--model_dir", type=str, required=True, help="模型目录路径")
    parser.add_argument("--prompt", type=str, required=True, help="推理提示词")
    parser.add_argument("--num_images", type=int, default=1, help="生成图像数量")
    parser.add_argument("--seed", type=int, default=None, help="随机种子")
    args = parser.parse_args()
    
    perform_inference(args.model_dir, args.prompt, num_images=args.num_images, seed=args.seed)
