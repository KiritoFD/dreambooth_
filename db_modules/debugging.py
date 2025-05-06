"""
DreamBooth 调试和错误分析模块
负责分析训练失败原因并提供解决方案
"""
import os
import datetime
import torch
import traceback
import psutil
import platform
from pathlib import Path


def analyze_training_failure(error, model_path, total_steps):
    """分析训练失败原因并提供解决方案建议"""
    print("\n" + "="*80)
    print("【训练失败分析】".center(80))
    print("="*80)
    
    # 检测常见错误类型
    error_str = str(error)
    gpu_info = get_gpu_info()
    
    if "CUDA out of memory" in error_str or "OOM" in error_str:
        print("检测到CUDA内存不足错误！")
        
        # 显示GPU内存使用情况
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"GPU总内存: {gpu_info['total_memory']:.1f}GB")
            print(f"当前使用内存: {gpu_info['used_memory']:.1f}GB")
            
            # 估计需要多少内存
            needed = estimate_memory_requirement(model_path)
            if needed:
                print(f"估计训练需要至少 {needed:.1f}GB GPU内存")
                if needed > gpu_info['total_memory']:
                    print(f"您的GPU内存不足，建议采用以下优化措施:")
                    print_memory_optimization_suggestions(gpu_info['total_memory'])
        
        # 创建详细调试报告
        report_path = create_debug_report(model_path, total_steps, error)
        print(f"\n详细调试报告已保存至: {report_path}")
    
    elif "Expected all tensors to be on the same device" in error_str:
        print("检测到设备不匹配错误")
        print("\n问题原因: 模型的不同部分可能位于不同设备（CPU/GPU）上")
        print("解决方案:")
        print("1. 确保所有模型组件移至同一设备")
        print("2. 检查代码中是否明确指定了设备")
        print("3. 尝试重启训练")
    
    elif "CUDA error: device-side assert triggered" in error_str:
        print("检测到CUDA设备断言错误")
        print("\n问题原因: 可能是数据格式问题或索引越界")
        print("解决方案:")
        print("1. 检查输入数据的形状和类型")
        print("2. 设置环境变量 CUDA_LAUNCH_BLOCKING=1 以获得更详细的错误信息")
        print("3. 尝试更小的batch size")
    
    else:
        print(f"未识别的错误类型: {error_str[:200]}...")
        print("\n可能的解决方案:")
        print("1. 检查输入数据和路径")
        print("2. 尝试重新安装依赖")
        print("3. 降低训练参数（批次大小、先验图像数量等）")
    
    # 保存错误日志
    save_error_log(error, model_path)
    
    return True


def create_debug_report(model_path, total_steps, error=None):
    """创建详细的训练调试报告"""
    os.makedirs(model_path, exist_ok=True)
    debug_report_path = os.path.join(model_path, "debug_report.txt")
    
    # 获取已完成的步骤
    completed_steps = 0
    checkpoint_path = os.path.join(model_path, "checkpoint.pt")
    if os.path.exists(checkpoint_path):
        try:
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            completed_steps = checkpoint.get("global_step", 0)
        except Exception as e:
            print(f"读取检查点失败: {e}")
    
    # 编写调试报告
    with open(debug_report_path, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("DreamBooth训练调试报告\n")
        f.write("=" * 80 + "\n\n")
        
        # 记录基本信息
        f.write(f"报告生成时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"训练进度: {completed_steps}/{total_steps} 步 ({completed_steps/total_steps*100:.1f}%)\n\n")
        
        # 系统信息
        f.write("系统信息:\n")
        f.write("-" * 40 + "\n")
        
        try:
            f.write(f"操作系统: {platform.system()} {platform.version()}\n")
            f.write(f"Python版本: {platform.python_version()}\n")
            
            # CPU信息
            f.write(f"CPU: {psutil.cpu_count(logical=False)} 物理核心, {psutil.cpu_count()} 逻辑核心\n")
            f.write(f"系统内存: 总量 {psutil.virtual_memory().total/1024**3:.2f}GB, "
                   f"可用 {psutil.virtual_memory().available/1024**3:.2f}GB\n")
        except:
            f.write("无法获取系统信息\n")
        
        # GPU信息
        f.write("\nGPU信息:\n")
        f.write("-" * 40 + "\n")
        
        gpu_info = get_gpu_info()
        if gpu_info['available']:
            f.write(f"GPU型号: {gpu_info['name']}\n")
            f.write(f"GPU总内存: {gpu_info['total_memory']:.2f}GB\n")
            f.write(f"GPU已分配内存: {gpu_info['used_memory']:.2f}GB\n")
            f.write(f"GPU预留内存: {gpu_info['reserved_memory']:.2f}GB\n")
            
            if hasattr(torch.cuda, 'get_device_capability'):
                capability = torch.cuda.get_device_capability(0)
                f.write(f"GPU计算能力: {capability[0]}.{capability[1]}\n")
        else:
            f.write("未检测到GPU\n")
        
        # 包版本信息
        f.write("\n依赖包版本:\n")
        f.write("-" * 40 + "\n")
        try:
            import pkg_resources
            for pkg in ['torch', 'diffusers', 'transformers', 'accelerate', 'bitsandbytes']:
                try:
                    version = pkg_resources.get_distribution(pkg).version
                    f.write(f"{pkg}: {version}\n")
                except:
                    f.write(f"{pkg}: 未安装\n")
        except:
            f.write("无法获取依赖包版本信息\n")
        
        # 检查检查点文件
        f.write("\n检查点文件分析:\n")
        f.write("-" * 40 + "\n")
        
        if os.path.exists(checkpoint_path):
            f.write(f"检查点文件存在: {checkpoint_path}\n")
            f.write(f"检查点文件大小: {os.path.getsize(checkpoint_path)/1024/1024:.2f}MB\n")
            
            try:
                checkpoint = torch.load(checkpoint_path, map_location="cpu")
                f.write(f"检查点包含的步数: {checkpoint.get('global_step', '未知')}\n")
                f.write(f"检查点内容: {', '.join(checkpoint.keys())}\n")
                
                # 模型大小分析
                if "unet" in checkpoint:
                    unet_params = sum(p.numel() for p in checkpoint["unet"].values())
                    f.write(f"UNet参数数量: {unet_params/1e6:.2f}M\n")
                
                if "text_encoder" in checkpoint:
                    text_encoder_params = sum(p.numel() for p in checkpoint["text_encoder"].values())
                    f.write(f"文本编码器参数数量: {text_encoder_params/1e6:.2f}M\n")
                
            except Exception as e:
                f.write(f"无法加载检查点文件: {str(e)}\n")
        else:
            f.write("未找到检查点文件\n")
        
        # 检查数据集
        instance_dir = os.path.join(Path(model_path).parent, "instance_images")
        if os.path.exists(instance_dir):
            f.write("\n实例图像目录分析:\n")
            f.write("-" * 40 + "\n")
            image_files = [f for f in os.listdir(instance_dir) 
                         if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            f.write(f"找到 {len(image_files)} 张实例图像\n")
            if len(image_files) < 3:
                f.write("警告: 实例图像数量少于推荐的3-5张\n")
            elif len(image_files) > 10:
                f.write("警告: 实例图像数量较多，可能不必要地增加训练难度\n")
        
        # 检查生成的类别图像
        class_dir = os.path.join(model_path, "class_images")
        if os.path.exists(class_dir):
            f.write("\n类别图像目录分析:\n")
            f.write("-" * 40 + "\n")
            class_files = [f for f in os.listdir(class_dir) 
                         if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            f.write(f"找到 {len(class_files)} 张生成的类别图像\n")
            
            if len(class_files) < 100:
                f.write("警告: 类别图像数量少于建议的200张，可能影响先验保留效果\n")
        
        # 检查可能的错误日志
        error_log_path = os.path.join(model_path, "error_log.txt")
        if os.path.exists(error_log_path):
            f.write("\n错误日志内容:\n")
            f.write("-" * 40 + "\n")
            try:
                with open(error_log_path, "r") as error_file:
                    f.write(error_file.read())
            except:
                f.write("无法读取错误日志\n")
        
        # 记录当前错误信息
        if error:
            f.write("\n当前错误信息:\n")
            f.write("-" * 40 + "\n")
            f.write(str(error) + "\n\n")
            f.write("错误堆栈跟踪:\n")
            f.write(traceback.format_exc())
        
        # 给出可能的原因分析和建议
        f.write("\n可能的中断原因分析:\n")
        f.write("-" * 40 + "\n")
        
        if completed_steps == 0:
            f.write("训练可能一开始就失败了，主要原因可能是:\n")
            f.write("1. 模型加载失败 - 检查模型路径和网络连接\n")
            f.write("2. 内存不足 - 尝试减少batch size或prior_images数量\n")
            f.write("3. 环境问题 - 检查CUDA版本与PyTorch版本的兼容性\n")
        elif completed_steps < total_steps * 0.1:  # 不到10%
            f.write("训练在早期阶段失败，主要原因可能是:\n")
            f.write("1. 内存逐渐耗尽 - 模型在训练过程中累积了过多内存\n")
            f.write("2. 数据问题 - 某些训练图像可能有问题\n")
            f.write("3. 优化器设置问题 - 学习率可能过高导致不稳定\n")
        else:
            f.write("训练在进行一段时间后中断，主要原因可能是:\n")
            f.write("1. 系统或用户中断 - 检查是否有系统休眠或电源管理问题\n")
            f.write("2. 长时间训练导致的硬件过热 - 检查GPU温度\n")
            f.write("3. 其他程序干扰 - 检查是否有其他程序抢占GPU资源\n")
        
        # 给出解决建议
        f.write("\n建议解决方案:\n")
        f.write("-" * 40 + "\n")
        f.write("1. 减少资源使用:\n")
        f.write("   - 降低prior_images数量 (--prior_images 5)\n")
        f.write("   - 不训练文本编码器 (移除--train_text_encoder)\n")
        f.write("   - 减小batch_size (--batch_size 1)\n")
        f.write("   - 添加 --low_memory 参数启用内存优化\n")
        f.write("2. 恢复训练:\n")
        f.write("   - 使用--resume参数从上次中断处继续\n")
        f.write("3. 检查环境:\n")
        f.write("   - 更新GPU驱动\n")
        f.write("   - 确保CUDA版本与PyTorch兼容\n")
        f.write("   - 关闭其他占用GPU的程序\n")
    
    print(f"调试报告已生成: {debug_report_path}")
    return debug_report_path


def get_gpu_info():
    """获取GPU信息"""
    info = {
        'available': torch.cuda.is_available(),
        'name': 'N/A',
        'total_memory': 0,
        'reserved_memory': 0,
        'used_memory': 0,
        'free_memory': 0
    }
    
    if info['available']:
        info['name'] = torch.cuda.get_device_name(0)
        info['total_memory'] = torch.cuda.get_device_properties(0).total_memory / 1024**3
        info['reserved_memory'] = torch.cuda.memory_reserved(0) / 1024**3
        info['used_memory'] = torch.cuda.memory_allocated(0) / 1024**3
        info['free_memory'] = info['total_memory'] - info['used_memory']
    
    return info


def estimate_memory_requirement(model_path=None):
    """估计训练所需的GPU内存"""
    # 基础内存需求（基于经验值）
    base_memory = 3.0  # 基本模型加载约需3GB
    
    # 如果有检查点，可以从实际使用情况推测
    if model_path:
        checkpoint_path = os.path.join(model_path, "checkpoint.pt")
        if os.path.exists(checkpoint_path):
            try:
                # 检查点大小转为GB
                ckpt_size = os.path.getsize(checkpoint_path) / 1024**3
                # 实际运行时通常需要比检查点大小多2-3倍的内存
                return ckpt_size * 2.5
            except:
                pass
    
    # 预估值（单位：GB）
    unet_memory = 2.0  # U-Net
    text_encoder_memory = 1.0  # 文本编码器
    vae_memory = 0.5  # VAE
    optimizer_memory = 1.5  # 优化器状态
    batch_overhead = 1.0  # 批处理开销
    
    return base_memory + unet_memory + text_encoder_memory + vae_memory + optimizer_memory + batch_overhead


def print_memory_optimization_suggestions(available_memory):
    """根据可用内存打印优化建议"""
    if available_memory < 6:
        print("\n【极低内存优化建议 (< 6GB)】")
        print("1. 启用低内存模式: --low_memory")
        print("2. 禁用文本编码器训练 (不要使用--train_text_encoder)")
        print("3. 设置较低的先验图像数量: --prior_images 5")
        print("4. 使用较少的训练步数: --steps 500")
        print("5. 安装bitsandbytes进行8位优化")
    elif available_memory < 8:
        print("\n【低内存优化建议 (6-8GB)】")
        print("1. 启用低内存模式: --low_memory")
        print("2. 禁用文本编码器训练 (不要使用--train_text_encoder)")
        print("3. 使用适度的先验图像数量: --prior_images 10")
        print("4. 使用默认训练步数: --steps 10000")
    elif available_memory < 12:
        print("\n【中等内存优化建议 (8-12GB)】")
        print("1. 启用低内存模式: --low_memory")
        print("2. 可以尝试训练文本编码器: --train_text_encoder")
        print("3. 使用适当的先验图像数量: --prior_images 20")
    else:
        print("\n【高内存使用建议 (>12GB)】")
        print("1. 可以使用默认设置")
        print("2. 如需更好效果，可增加先验图像数量: --prior_images 50")
        print("3. 增加训练步数: --steps 1500")


def save_error_log(error, model_path):
    """保存错误日志到文件"""
    os.makedirs(model_path, exist_ok=True)
    error_log_path = os.path.join(model_path, "error_log.txt")
    
    with open(error_log_path, "w", encoding="utf-8") as f:
        f.write(f"错误时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"错误类型: {type(error).__name__}\n")
        f.write(f"错误描述: {str(error)}\n\n")
        f.write("调用栈跟踪:\n")
        f.write(traceback.format_exc())
        
        f.write("\n\n系统信息:\n")
        try:
            f.write(f"操作系统: {platform.system()} {platform.version()}\n")
            f.write(f"Python版本: {platform.python_version()}\n")
            f.write(f"PyTorch版本: {torch.__version__}\n")
            f.write(f"CUDA可用: {torch.cuda.is_available()}\n")
            if torch.cuda.is_available():
                f.write(f"CUDA版本: {torch.version.cuda}\n")
                f.write(f"GPU型号: {torch.cuda.get_device_name(0)}\n")
        except:
            f.write("无法获取系统信息\n")
    
    return error_log_path


if __name__ == "__main__":
    # 简单的测试代码
    print("DreamBooth 调试工具")
    print("系统信息:", platform.system(), platform.version())
    print("Python版本:", platform.python_version())
    if torch.cuda.is_available():
        print("GPU型号:", torch.cuda.get_device_name(0))
