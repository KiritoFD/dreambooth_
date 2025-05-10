"""
简单的图像生成脚本，使用高质量推理模块
"""
import os
import argparse
import json
import torch
from inference import load_pipeline_from_config, generate_high_quality_images, save_images

def main():
    parser = argparse.ArgumentParser(description="使用训练好的DreamBooth模型生成图像")
    parser.add_argument("--config", type=str, default="config.json", help="配置文件路径")
    parser.add_argument("--prompt", type=str, default=None, help="生成提示，默认使用配置中的实例提示")
    parser.add_argument("--num_images", type=int, default=4, help="要生成的图像数量")
    parser.add_argument("--seed", type=int, default=None, help="随机种子，用于可复现生成")
    parser.add_argument("--height", type=int, default=512, help="图像高度")
    parser.add_argument("--width", type=int, default=512, help="图像宽度")
    parser.add_argument("--output_dir", type=str, default=None, help="输出目录，默认为config中的output_dir/generated")
    
    args = parser.parse_args()
    
    # 加载配置
    with open(args.config, "r") as f:
        config = json.load(f)
    
    # 确定设备
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")
    
    # 设置提示
    if args.prompt is None:
        instance_token = config["dataset"]["instance_prompt"].split()[-1]
        args.prompt = f"A portrait of {instance_token} in a beautiful landscape, high quality, detailed"
        print(f"未提供提示，使用默认提示: {args.prompt}")
    
    # 设置输出目录
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = os.path.join(config["paths"]["output_dir"], "generated")
    
    # 加载模型
    pipe = load_pipeline_from_config(config, device)
    
    # 生成高质量图像
    print(f"开始生成 {args.num_images} 张图像, 提示: '{args.prompt}'")
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
    saved_paths = save_images(images, output_dir)
    
    print(f"生成完成! {len(saved_paths)} 张图像已保存到 {output_dir}")
    print("保存的图像路径:")
    for path in saved_paths:
        print(f" - {path}")

if __name__ == "__main__":
    main()
