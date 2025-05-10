import os
import gc
import torch
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
    print("版本: (Config-Driven - Default to Training)") # Updated versioning note
    
    missing_deps = check_dependencies()
    if missing_deps:
        print(f"错误: 缺少必要的依赖项: {', '.join(missing_deps)}")
        print(f"请运行: pip install {' '.join(missing_deps)}")
        return 1

    # No longer using argparse, config file is fixed
    config_file_path = "config.json"

    if not os.path.exists(config_file_path):
        print(f"错误: 配置文件 '{config_file_path}' 未找到.")
        return 1
    
    config_data = load_config(config_file_path)

    # Directly proceed to training
    print(f"开始训练，所有参数从 '{config_file_path}' 加载...")
    
    # 调试：打印数据集路径配置
    dataset_config = config_data.get("dataset", {})
    instance_data_dir = dataset_config.get("instance_data_dir")
    class_data_dir = dataset_config.get("class_data_dir")
    print(f"[DEBUG] 从配置加载的实例图片路径: {instance_data_dir}")
    if class_data_dir:
        print(f"[DEBUG] 从配置加载的类别图片路径: {class_data_dir}")
    else:
        print("[DEBUG] 未配置类别图片路径 (如果使用了先验保留，则会自动生成或需要指定)。")

    result = dreambooth_training(config_data)
    if result is None:
        identifier, training_successful = None, False
    else:
        identifier, training_successful = result
        
    if training_successful:
        print(f"训练成功完成。标识符: {identifier}")
    else:
        print(f"训练未成功完成或被中断。标识符: {identifier}")
    
    return 0

if __name__ == "__main__":
    exit(main())
