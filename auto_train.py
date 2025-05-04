"""
DreamBooth 简化自动训练脚本
直接执行100次训练，无需任何用户交互
直接调用main.py中的函数，而不是通过命令行
"""
import os
import sys
import time
import gc
from datetime import datetime
import torch

# 导入main.py中的函数
from main import dreambooth_training, download_small_model, MemoryManager

def auto_train(iterations=100):
    """
    直接调用dreambooth_training函数执行多次训练
    
    Args:
        iterations: 重复执行的次数
    """
    # 训练参数设置
    params = {
        "instance_data_dir": "./assets",
        "class_prompt": "a witch",
        "low_memory": True,
        "prior_images": 5,
        "resume": True,
        "output_dir": "./output",
        "model_name": None,  # 将使用默认的小型模型
        "steps": 1000,
        "prior_weight": 1.0,
        "train_text_encoder": False,
        "batch_size": 1
    }
    
    # 记录开始时间
    start_time = datetime.now()
    print(f"\n" + "="*80)
    print(f"DreamBooth 自动训练开始")
    print(f"计划执行 {iterations} 次训练")
    print(f"开始时间: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    
    # 检查实例图像目录是否存在
    if not os.path.exists(params["instance_data_dir"]):
        print(f"错误: 实例图像目录 '{params['instance_data_dir']}' 不存在!")
        return False
    
    # 确保输出目录存在
    os.makedirs(params["output_dir"], exist_ok=True)
    
    # 如果model_name为None，获取默认小型模型
    if params["model_name"] is None:
        params["model_name"] = download_small_model()
    
    # 内存优化参数
    if params["low_memory"]:
        try:
            from db_modules.memory_optimization import get_optimal_settings
            if torch.cuda.is_available():
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
                config = get_optimal_settings(gpu_memory)
                
                print(f"\n应用内存优化: 针对{gpu_memory:.1f}GB GPU")
                
                # 低内存训练参数
                low_memory_params = {
                    "attention_slice_size": config.get("attention_slice_size", 0),
                    "gradient_checkpointing": config.get("gradient_checkpointing", True),
                    "use_8bit_adam": config.get("use_8bit_adam", True)
                }
            else:
                low_memory_params = {}
        except ImportError:
            print("未找到内存优化模块，将使用标准设置")
            low_memory_params = {}
    else:
        low_memory_params = {}
    
    # 执行训练循环
    for i in range(iterations):
        # 计算进度和时间
        curr_time = datetime.now()
        elapsed = (curr_time - start_time).total_seconds() / 60.0
        eta = (elapsed / (i + 1)) * (iterations - i - 1) if i > 0 else 0
        
        print(f"\n" + "-"*80)
        print(f"[{i+1}/{iterations}] 开始训练回合")
        print(f"已用时: {elapsed:.1f}分钟 - 预计剩余: {eta:.1f}分钟")
        print(f"当前时间: {curr_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print("-"*80)
        
        # 创建内存管理器
        memory_mgr = MemoryManager()
        
        # 强制清理内存
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        try:
            # 直接调用训练函数
            identifier, training_successful = dreambooth_training(
                pretrained_model_name_or_path=params["model_name"],
                instance_data_dir=params["instance_data_dir"],
                output_dir=params["output_dir"],
                class_prompt=params["class_prompt"],
                max_train_steps=params["steps"],
                prior_preservation_weight=params["prior_weight"],
                train_text_encoder=params["train_text_encoder"],
                prior_generation_samples=params["prior_images"],
                train_batch_size=params["batch_size"],
                memory_mgr=memory_mgr,
                resume_training=params["resume"],
                **low_memory_params
            )
            
            print(f"\n训练回合 {i+1} 完成: {'成功' if training_successful else '未成功完成'}")
            
        except Exception as e:
            print(f"\n训练回合 {i+1} 出错: {str(e)}")
            import traceback
            traceback.print_exc()
        
        # 训练完成后短暂休息，防止系统过热
        print(f"等待10秒后开始下一回合...")
        time.sleep(10)
    
    # 总结
    end_time = datetime.now()
    total_elapsed = (end_time - start_time).total_seconds() / 3600.0
    print("\n" + "="*80)
    print(f"自动训练完成!")
    print(f"开始时间: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"结束时间: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"总用时: {total_elapsed:.2f}小时")
    print("="*80)
    
    return True

if __name__ == "__main__":
    auto_train(iterations=100)