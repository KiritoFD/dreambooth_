import os
import sys
import argparse
import json
import torch
from dreambooth import dreambooth_training
from db_modules.inference_utils import load_inference_pipeline, generate_images

def check_dependencies(): # Keep as a utility
    """检查必要的依赖项是否已安装"""
    missing_deps = []
    try:
        import torch
    except ImportError:
        missing_deps.append("torch")
    
    try:
        import accelerate
    except ImportError:
        missing_deps.append("accelerate")
        
    try:
        import transformers
        transformers_version = getattr(transformers, '__version__', '0.0.0')
        if transformers_version < '4.20.0':
            print(f"transformers=={transformers_version} (建议 >= 4.20.0)")
    except ImportError:
        missing_deps.append("transformers")
    
    try:
        import diffusers
    except ImportError:
        missing_deps.append("diffusers")
    
    return missing_deps

def load_config(config_path): # Helper function, same as in dreambooth.py
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config

def main():
    parser = argparse.ArgumentParser(description="DreamBooth 训练与推理工具")
    parser.add_argument("--config", type=str, default="config.json", help="配置文件路径")
    
    # 添加推理相关参数
    parser.add_argument("--infer", action="store_true", help="进行推理模式而非训练模式")
    parser.add_argument("--prompt", type=str, default=None, help="推理时的提示语")
    parser.add_argument("--negative_prompt", type=str, default=None, help="推理时的否定提示语")
    parser.add_argument("--num_images", type=int, default=4, help="要生成的图像数量")
    parser.add_argument("--seed", type=int, default=None, help="随机种子")
    parser.add_argument("--height", type=int, default=512, help="生成图像高度")
    parser.add_argument("--width", type=int, default=512, help="生成图像宽度")
    parser.add_argument("--output_dir", type=str, default=None, help="生成图像的输出目录")
    
    args = parser.parse_args()
    
    # 检查配置文件是否存在
    if not os.path.exists(args.config):
        print(f"错误: 配置文件 '{args.config}' 不存在")
        return 1
    
    # 加载配置
    with open(args.config, "r") as f:
        config_data = json.load(f)
    
    # 确保配置中包含inference部分
    if "inference" not in config_data:
        config_data["inference"] = {
            "num_inference_steps": 50,
            "guidance_scale": 7.5,
            "negative_prompt": "ugly, blurry, poor quality, deformed, disfigured, malformed limbs, missing limbs, bad anatomy, bad proportions",
            "scheduler": "DPMSolverMultistep",
            "use_karras_sigmas": True,
            "noise_offset": 0.1,
            "vae_batch_size": 1,
            "high_quality_mode": True
        }
    
    if args.infer:
        # 推理模式
        print("启动推理模式...")
        
        # 确定设备
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"使用设备: {device}")
        
        # 加载推理管道
        pipe, model_path, model_info = load_inference_pipeline(config_data, device)
        print(f"使用模型: {model_info}")
        
        # 设置提示语
        if args.prompt is None:
            # 如果未提供提示语，使用实例提示加上一些修饰词
            instance_token = config_data["dataset"]["instance_prompt"].split()[-1]
            # 提取实例提示中除了最后一个词（通常是特定标识符）之外的内容
            category = ' '.join(config_data["dataset"]["instance_prompt"].split()[:-1]).strip('a ').strip('an ')
            args.prompt = f"a {instance_token} {category}, masterpiece, highly detailed, high quality"
            print(f"未提供提示语，使用默认提示: {args.prompt}")
        
        # 生成图像
        saved_paths = generate_images(
            pipe,
            args.prompt,
            config_data,
            negative_prompt=args.negative_prompt,
            height=args.height,
            width=args.width,
            num_images=args.num_images,
            seed=args.seed,
            output_dir=args.output_dir
        )
        
        # 显示生成的图像路径
        print("\n生成的图像:")
        for i, path in enumerate(saved_paths):
            print(f"{i+1}. {path}")
        
        return 0
    else:
        # 训练模式
        print("启动训练模式...")
        identifier, training_successful = dreambooth_training(config_data)
        
        if training_successful:
            print(f"DreamBooth 训练成功完成，标识符: {identifier}")
            return 0
        else:
            print(f"DreamBooth 训练失败或被中断，标识符: {identifier}")
            return 1

if __name__ == "__main__":
    exit(main())
