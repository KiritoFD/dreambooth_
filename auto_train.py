"""
DreamBooth 简化自动训练脚本
通过加载配置文件执行训练。
"""
import os
# import sys # Not used directly
import time
import gc
from datetime import datetime, timedelta  # 正确导入 timedelta
import torch
import json # For loading config
import argparse # For specifying config file

# 导入main.py中的函数 (dreambooth_training is now the primary one)
from dreambooth import dreambooth_training 
# from main import download_small_model, MemoryManager # Keep if needed for other logic, but training is via config
# from db_modules.model_loader import find_local_model, load_model_with_local_priority # Model path is now in config

def load_config(config_path): # Helper
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config

def auto_train_entry(config_file_path, iterations=200): # iterations can be part of config too
    """
    自动训练入口点，加载配置并执行训练。
    
    Args:
        config_file_path (str): 配置文件的路径。
        iterations (int): 重复执行的次数 (如果希望多次运行相同配置)。
    """
    if not os.path.exists(config_file_path):
        print(f"错误: 配置文件 '{config_file_path}' 不存在!")
        return False
        
    config_data = load_config(config_file_path)
    
    # iterations can also be a field in config_data if desired
    # num_iterations = config_data.get("auto_train_iterations", iterations)

    print(f"\n" + "="*80)
    print(f"DreamBooth 自动训练开始 (Config-Driven)")
    iterations +=200
    print(f"计划执行 {iterations} 次训练迭代使用配置: {config_file_path}")
    start_time = datetime.now()
    print(f"开始时间: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    
    # 实例图像目录检查 (从config获取)
    instance_dir = config_data.get("paths", {}).get("instance_data_dir", None)
    if not instance_dir or not os.path.exists(instance_dir):
        print(f"错误: 实例图像目录 '{instance_dir}' (来自config) 不存在或未在config中定义!")
        return False
    
    # 输出目录检查 (从config获取)
    output_dir_config = config_data.get("paths", {}).get("output_dir", "./output_auto")
    os.makedirs(output_dir_config, exist_ok=True) # Ensure output_dir from config exists
    config_data["paths"]["output_dir"] = output_dir_config # Ensure it's set for the training call

    # memory_mgr = MemoryManager(config_data.get("memory_optimization", {}).get("aggressive_gc", False)) # dreambooth_training handles its own

    all_iterations_successful = True
    for i in range(iterations):
        curr_time = datetime.now()
        elapsed_total = (curr_time - start_time).total_seconds()
        
        print(f"\n" + "-"*80)
        print(f"自动训练迭代 [{i+1}/{iterations}]")
        if i > 0:
            avg_time_per_iter = elapsed_total / i
            eta_seconds = avg_time_per_iter * (iterations - i)
            eta_str = str(timedelta(seconds=int(eta_seconds)))
            print(f"已用时: {str(timedelta(seconds=int(elapsed_total)))} - 预计剩余: {eta_str}")
        print(f"当前时间: {curr_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"使用模型 (来自config): {config_data.get('paths', {}).get('pretrained_model_name_or_path')}")
        print("-"*80)
        
        # 强制清理内存 (dreambooth_training might do this too via MemoryManager)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        try:
            # 直接调用训练函数
            # Note: If iterations > 1 and output_dir is the same, checkpoints will overwrite.
            # Consider modifying output_dir per iteration if separate outputs are needed.
            # For now, using the same config means it will try to resume or overwrite.
            
            # If each iteration should be fresh, ensure resume_from_checkpoint is null or handle output_dir
            current_config = config_data.copy() # Use a copy if modifications per iteration are needed
            if iterations > 1:
                 # Example: make output_dir unique per iteration
                 # current_config["paths"]["output_dir"] = os.path.join(output_dir_config, f"iter_{i+1}")
                 # os.makedirs(current_config["paths"]["output_dir"], exist_ok=True)
                 # current_config["training"]["resume_from_checkpoint"] = None # Ensure fresh start unless intended
                 pass


            identifier, training_successful_iter = dreambooth_training(current_config)
            
            print(f"\n训练迭代 {i+1} 完成: {'成功' if training_successful_iter else '未成功完成'}")
            if not training_successful_iter:
                all_iterations_successful = False
            
        except Exception as e:
            print(f"\n训练迭代 {i+1} 出错: {str(e)}")
            import traceback
            traceback.print_exc()
            all_iterations_successful = False
            # Decide if to continue next iteration or stop
            # break 
        
        if i < iterations - 1:
            print(f"等待10秒后开始下一迭代...")
            time.sleep(10) # Prevent system overheat or API rate limits if downloading
    
    end_time = datetime.now()
    total_duration = end_time - start_time
    print("\n" + "="*80)
    print(f"所有自动训练迭代完成")
    print(f"总用时: {str(total_duration)}")
    print(f"结束时间: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    return all_iterations_successful

if __name__ == "__main__":
    cli_parser = argparse.ArgumentParser(description="DreamBooth 自动训练脚本 (Config-Driven)")
    cli_parser.add_argument(
        "--config_file",
        type=str,
        default="config.json",
        help="Path to the training configuration JSON file.",
    )
    cli_parser.add_argument(
        "--iterations",
        type=int,
        default=1, # Default to 1 iteration if not specified
        help="Number of training iterations to run with the same config."
    )
    cli_args = cli_parser.parse_args()

    auto_train_entry(config_file_path=cli_args.config_file, iterations=cli_args.iterations)