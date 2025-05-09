import os
import gc
import torch
import argparse
import datetime
import json # Added for loading config
from dreambooth import dreambooth_training # Assuming dreambooth_training is in dreambooth.py

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
    print("\n【DreamBooth训练/推理工具】")
    print("版本: (Config-Driven)") # Updated versioning note
    
    missing_deps = check_dependencies()
    if missing_deps:
        print(f"错误: 缺少必要的依赖项: {', '.join(missing_deps)}")
        print(f"请运行: pip install {' '.join(missing_deps)}")
        return 1

    parser = argparse.ArgumentParser(description="DreamBooth 训练和推理 (Config-Driven)")
    parser.add_argument(
        "--config_file",
        type=str,
        default="config.json",
        help="Path to the configuration JSON file for training or inference.",
    )
    parser.add_argument("--train", action="store_true", help="执行训练模式 (参数来自config_file)")
    parser.add_argument("--infer", action="store_true", help="执行推理模式 (参数来自config_file or specific args)")
    parser.add_argument("--prompt", type=str, help="[推理模式] 推理时的提示词 (可覆盖config中的validation_prompt)")
    parser.add_argument("--num_images", type=int, default=1, help="[推理模式] 生成图像数量")
    parser.add_argument("--seed", type=int, help="[推理模式] 随机种子，用于可复现结果")

    args = parser.parse_args()

    if not os.path.exists(args.config_file):
        print(f"错误: 配置文件 '{args.config_file}' 未找到.")
        return 1
    
    config_data = load_config(args.config_file)

    if args.train:
        print(f"开始训练，所有参数从 '{args.config_file}' 加载...")
        
        # 调试：打印数据集路径配置
        dataset_config = config_data.get("dataset", {})
        instance_data_dir = dataset_config.get("instance_data_dir")
        class_data_dir = dataset_config.get("class_data_dir")
        print(f"[DEBUG] 从配置加载的实例图片路径: {instance_data_dir}")
        if class_data_dir:
            print(f"[DEBUG] 从配置加载的类别图片路径: {class_data_dir}")
        else:
            print("[DEBUG] 未配置类别图片路径 (如果使用了先验保留，则会自动生成或需要指定)。")

        identifier, training_successful = dreambooth_training(config_data)
            
        if training_successful:
            print(f"训练成功完成。标识符: {identifier}")
        else:
            print(f"训练未成功完成或被中断。标识符: {identifier}")

    elif args.infer:
        print(f"开始推理，参数主要从 '{args.config_file}' 加载...")
        infer_prompt = args.prompt if args.prompt else config_data.get("logging_saving", {}).get("validation_prompt", "a photo of a sks dog")
        output_model_path = config_data["paths"]["output_dir"]
        
        if not os.path.exists(output_model_path):
            print(f"错误: 推理所需的模型目录 '{output_model_path}' (来自config) 不存在。请先训练模型。")
            return 1
            
        print(f"将使用模型目录: {output_model_path}")
        print(f"提示词: {infer_prompt}")
        
        # 调用推理功能
        try:
            from inference import perform_inference          
            num_images = args.num_images
            seed = args.seed
            
            # 传递完整的配置数据
            image_paths = perform_inference(
                model_dir=output_model_path,
                prompt=infer_prompt,
                num_images=num_images,
                seed=seed,
                config=config_data  # 传递完整的配置
            )
            
            if image_paths:
                print(f"\n推理完成！")
                print(f"生成了 {len(image_paths)} 张图像，保存在: {os.path.dirname(image_paths[0])}")
            else:
                print(f"\n推理失败，未生成图像。请查看上方错误信息。")
            
        except ImportError:
            print("错误: 未能导入推理模块。请确保 'inference.py' 文件存在。")
            return 1
        except Exception as e:
            print(f"执行推理时发生错误: {str(e)}")
            import traceback
            traceback.print_exc()
            return 1

    else:
        print("请指定操作模式: --train 或 --infer")
        print(f"示例: python main.py --train --config_file my_custom_config.json")
        print(f"示例: python main.py --infer --config_file my_custom_config.json --prompt \"a photo of sks person\"")
    
    return 0

if __name__ == "__main__":
    exit(main())
