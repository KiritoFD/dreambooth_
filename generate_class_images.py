"""
类别图像生成脚本
用于生成用于Dreambooth训练的先验类别图像
"""
import os
import argparse
import torch
from diffusers import StableDiffusionPipeline
from db_modules.prior_generation import generate_prior_images

def get_unique_folder_name(base_path):
    """
    生成唯一的文件夹名称。如果文件夹已存在，则在名称后添加数字
    """
    if not os.path.exists(base_path):
        return base_path
        
    folder_name = os.path.basename(base_path)
    parent_dir = os.path.dirname(base_path)
    
    counter = 1
    while os.path.exists(base_path):
        new_name = f"{folder_name}_{counter}"
        base_path = os.path.join(parent_dir, new_name)
        counter += 1
        
    return base_path

def main(args):
    # 设置默认提示词和输出目录
    prompt = args.prompt
    output_dir = args.output_dir
    num_images = args.num_images
    batch_size = args.batch_size
    
    # 确保输出目录不会覆盖现有目录
    if os.path.exists(output_dir) and not args.overwrite:
        output_dir = get_unique_folder_name(output_dir)
        print(f"输出目录已存在，使用新路径: {output_dir}")
    
    print(f"准备生成 {num_images} 张图像，提示词: '{prompt}'")
    print(f"输出目录: {output_dir}")
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 设置设备
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")
    
    # 加载模型
    print("加载Stable Diffusion模型...")
    try:
        pipeline = StableDiffusionPipeline.from_pretrained(
            args.model_id,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32
        )
        pipeline = pipeline.to(device)
        
        # 设置内存优化
        if device == "cuda":
            pipeline.enable_attention_slicing()
            if args.enable_xformers:
                try:
                    pipeline.enable_xformers_memory_efficient_attention()
                    print("已启用xformers内存优化")
                except:
                    print("无法启用xformers，将使用默认优化")
    except Exception as e:
        print(f"加载模型时出错: {e}")
        return
    
    # 生成类别图像
    class_images_dir = os.path.join(output_dir, "class_images")
    print(f"\n开始生成类别图像到: {class_images_dir}")
    
    generate_prior_images(
        pipeline=pipeline,
        class_prompt=prompt,
        output_dir=output_dir,
        num_samples=num_images,
        batch_size=batch_size,
        class_images_dir=class_images_dir
    )
    
    # 检查生成结果
    image_count = len([f for f in os.listdir(class_images_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    print(f"\n生成完成！共生成 {image_count} 张类别图像，保存在 {class_images_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="生成用于Dreambooth训练的类别图像")
    parser.add_argument("--prompt", type=str, default="a furry rootic monkey", help="生成图像的提示词")
    parser.add_argument("--output_dir", type=str, default="outputs/class_images", help="输出目录")
    parser.add_argument("--num_images", type=int, default=200, help="要生成的图像数量")
    parser.add_argument("--batch_size", type=int, default=None, help="批处理大小，如不指定将根据GPU内存自动设置")
    parser.add_argument("--model_id", type=str, default="runwayml/stable-diffusion-v1-5", help="要使用的模型ID")
    parser.add_argument("--enable_xformers", action="store_true", help="启用xformers内存优化")
    parser.add_argument("--overwrite", action="store_true", help="覆盖现有输出目录")
    
    args = parser.parse_args()
    main(args)
