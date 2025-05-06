"""
DreamBooth 简化自动训练脚本
直接执行训练，无需任何用户交互
实现典型的DreamBooth训练流程，优先使用本地模型
"""
import os
import sys
import time
import gc
from datetime import datetime
import torch

# 导入main.py中的函数
from main import dreambooth_training, download_small_model, MemoryManager
# 导入本地模型加载模块
from db_modules.model_loader import find_local_model, load_model_with_local_priority

def auto_train(iterations=100):
    """
    直接调用dreambooth_training函数执行多次训练
    
    Args:
        iterations: 重复执行的次数
    """
    # 训练参数设置 - 根据DreamBooth标准训练流程调整
    params = {
        # 基础训练参数
        "instance_data_dir": "./assets",
        "instance_prompt": "a photo of a sks container",  # 使用唯一标识符
        "class_prompt": "a photo of a container",  # 一般类别名称
        "output_dir": "./output",
        "steps": 800,  # DreamBooth训练通常800-1500步
        
        # 先验保留相关参数
        "prior_preservation": True,
        "prior_weight": 1.0,
        "prior_images": 100,  # 典型值为100
        
        # 优化参数
        "learning_rate": 5e-6,  # 典型学习率
        "train_text_encoder": True,  # 通常为True以获得更好效果
        "batch_size": 1,
        "gradient_accumulation": 1,
        
        # 本地模型和恢复训练
        "model_name": "CompVis/stable-diffusion-v1-4",  # 默认模型
        "use_local_model": True,  # 优先使用本地模型
        "custom_model_path": None,  # 自定义本地模型路径
        "resume": True,
        
        # 内存优化
        "low_memory": True,
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
    
    # 优先使用本地模型
    if params["use_local_model"]:
        print("\n正在查找本地模型...")
        
        # 如果提供了自定义路径，首先检查该路径
        if params["custom_model_path"] and os.path.exists(params["custom_model_path"]):
            local_model_path = params["custom_model_path"]
            print(f"使用指定的本地模型: {local_model_path}")
        else:
            # 查找本地模型
            local_model_path = find_local_model(params["model_name"])
            if local_model_path:
                print(f"找到本地模型: {local_model_path}")
                params["model_name"] = local_model_path  # 使用本地模型路径
            else:
                print(f"未找到本地模型 '{params['model_name']}'，将尝试下载")
    
    # 添加实例损失监控参数
    params["monitor_loss"] = True
    params["instance_loss_weight"] = 1.0  # 确保实例损失有足够的权重
    
    # 内存优化参数
    if params["low_memory"]:
        try:
            from db_modules.memory_optimization import get_optimal_settings
            if torch.cuda.is_available():
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
                config = get_optimal_settings(gpu_memory)
                
                print(f"\n应用内存优化: 针对{gpu_memory:.1f}GB GPU")
                
                # 低内存训练参数 - 调整以避免实例损失为0
                low_memory_params = {
                    "attention_slice_size": config.get("attention_slice_size", 4),
                    "gradient_checkpointing": config.get("gradient_checkpointing", True),
                    "use_8bit_adam": config.get("use_8bit_adam", True),
                    "mixed_precision": config.get("mixed_precision", "fp16"),
                    "enable_xformers": config.get("enable_xformers", False),
                    "scale_lr": True,  # 根据批次大小缩放学习率
                }
            else:
                low_memory_params = {}
        except ImportError:
            print("未找到内存优化模块，将使用标准设置")
            low_memory_params = {}
    else:
        low_memory_params = {}
    
    # 创建内存管理器
    memory_mgr = MemoryManager()
    
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
        print(f"使用模型: {params['model_name']}")
        print("-"*80)
        
        # 强制清理内存
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        try:
            # 直接调用训练函数，使用更多DreamBooth特定参数
            identifier, training_successful = dreambooth_training(
                pretrained_model_name_or_path=params["model_name"],
                instance_data_dir=params["instance_data_dir"],
                output_dir=params["output_dir"],
                instance_prompt=params["instance_prompt"],
                class_prompt=params["class_prompt"],
                learning_rate=params["learning_rate"],
                max_train_steps=params["steps"],
                prior_preservation=params["prior_preservation"],
                prior_preservation_weight=params["prior_weight"],
                train_text_encoder=params["train_text_encoder"],
                prior_generation_samples=params["prior_images"],
                train_batch_size=params["batch_size"],
                gradient_accumulation_steps=params["gradient_accumulation"],
                memory_mgr=memory_mgr,
                resume_training=params["resume"],
                use_local_models=params["use_local_model"],
                local_model_path=local_model_path if "local_model_path" in locals() else None,
                **low_memory_params
            )
            
            print(f"\n训练回合 {i+1} 完成: {'成功' if training_successful else '未成功完成'}")
            
            # 更新输出路径以包含标识符，便于后续训练
            if training_successful and identifier:
                params["output_dir"] = os.path.join("./output", identifier)
                print(f"后续训练将保存到: {params['output_dir']}")
            
        except Exception as e:
            print(f"\n训练回合 {i+1} 出错: {str(e)}")
            import traceback
            traceback.print_exc()
            
            # 如果是模型加载错误，尝试重新设置模型来源
            if "load_config" in str(e) or "SSL" in str(e) or "download" in str(e):
                print("\n检测到模型加载错误，尝试使用备用模型...")
                
                # 尝试使用小模型
                try:
                    backup_model = download_small_model()
                    if backup_model:
                        params["model_name"] = backup_model
                        print(f"将使用备用模型: {backup_model}")
                except:
                    print("无法下载备用模型，请检查网络连接或提供本地模型路径")
        
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