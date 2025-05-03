import os
import time
import datetime
import gc
import torch
import psutil
import platform

class DebugMonitor:
    """训练进程监控和调试工具"""
    def __init__(self, output_dir, log_interval=50):
        self.output_dir = output_dir
        self.log_interval = log_interval
        self.start_time = time.time()
        self.last_log_time = self.start_time
        self.step_times = []
        self.memory_usages = []
        self.log_file = os.path.join(output_dir, "training_debug.log")
        self.step_durations = []
        
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)
        
        # 初始化日志文件
        with open(self.log_file, "w") as f:
            f.write(f"DreamBooth训练调试日志 - 开始时间: {datetime.datetime.now()}\n")
            f.write("-" * 80 + "\n")
            f.write(f"系统: {platform.system()} {platform.version()}\n")
            f.write(f"Python: {platform.python_version()}\n")
            
            if torch.cuda.is_available():
                f.write(f"GPU: {torch.cuda.get_device_name(0)}\n")
                f.write(f"CUDA版本: {torch.version.cuda}\n")
                f.write(f"GPU内存总量: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB\n")
            else:
                f.write("未检测到GPU\n")
                
            f.write(f"CPU: {psutil.cpu_count(logical=False)} 物理核心, {psutil.cpu_count()} 逻辑核心\n")
            f.write(f"内存: {psutil.virtual_memory().total / 1024**3:.2f} GB\n")
            f.write("-" * 80 + "\n\n")
    
    def log_step(self, step, total_steps, loss_dict=None, force=False):
        """记录训练步骤的状态"""
        current_time = time.time()
        
        # 只有当达到记录间隔或强制记录时才记录
        if not force and step % self.log_interval != 0:
            return
            
        # 计算步骤时间
        if self.step_times:
            step_duration = (current_time - self.last_log_time) / (step - self.step_times[-1][0])
            self.step_durations.append(step_duration)
            
            # 估计剩余时间
            avg_duration = sum(self.step_durations[-10:]) / min(len(self.step_durations), 10)
            remaining_steps = total_steps - step
            remaining_time = avg_duration * remaining_steps
            
            # 格式化为小时:分钟:秒
            hours, remainder = divmod(remaining_time, 3600)
            minutes, seconds = divmod(remainder, 60)
            eta = f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}"
        else:
            eta = "计算中..."
            step_duration = 0
        
        # 获取内存使用情况
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.memory_allocated() / 1024**3
            gpu_max_memory = torch.cuda.max_memory_allocated() / 1024**3
            self.memory_usages.append(gpu_memory)
        else:
            gpu_memory = 0
            gpu_max_memory = 0
            
        cpu_memory = psutil.Process(os.getpid()).memory_info().rss / 1024**3
        
        # 记录到日志文件
        with open(self.log_file, "a") as f:
            f.write(f"步骤 {step}/{total_steps} ({step/total_steps*100:.1f}%) - ETA: {eta}\n")
            f.write(f"每步时间: {step_duration:.2f}秒\n")
            f.write(f"GPU内存: {gpu_memory:.2f}GB (峰值: {gpu_max_memory:.2f}GB)\n")
            f.write(f"CPU内存: {cpu_memory:.2f}GB\n")
            
            if loss_dict:
                f.write("损失值:\n")
                for k, v in loss_dict.items():
                    f.write(f"  {k}: {v}\n")
            
            f.write("\n")
            
        # 更新上一次日志时间和步骤
        self.last_log_time = current_time
        self.step_times.append((step, current_time))
        
    def log_error(self, error, step, total_steps):
        """记录错误信息"""
        with open(self.log_file, "a") as f:
            f.write("\n" + "!" * 80 + "\n")
            f.write(f"错误发生在步骤 {step}/{total_steps} ({step/total_steps*100:.1f}%)\n")
            f.write(f"错误时间: {datetime.datetime.now()}\n")
            f.write(f"错误信息: {str(error)}\n")
            
            # 记录额外的系统状态
            if torch.cuda.is_available():
                f.write(f"当前GPU内存: {torch.cuda.memory_allocated() / 1024**3:.2f}GB\n")
                f.write(f"最大GPU内存: {torch.cuda.max_memory_allocated() / 1024**3:.2f}GB\n")
            
            f.write(f"当前CPU内存: {psutil.Process(os.getpid()).memory_info().rss / 1024**3:.2f}GB\n")
            f.write(f"系统内存使用率: {psutil.virtual_memory().percent}%\n")
            f.write("!" * 80 + "\n")
        
    def log_completion(self, step, total_steps, success=True):
        """记录训练完成情况"""
        duration = time.time() - self.start_time
        hours, remainder = divmod(duration, 3600)
        minutes, seconds = divmod(remainder, 60)
        
        with open(self.log_file, "a") as f:
            f.write("\n" + "=" * 80 + "\n")
            if success and step >= total_steps:
                f.write("训练成功完成!\n")
            else:
                f.write(f"训练在 {step}/{total_steps} ({step/total_steps*100:.1f}%) 步提前停止\n")
                
            f.write(f"总时长: {int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}\n")
            f.write(f"平均每步时间: {duration/max(step, 1):.2f}秒\n")
            
            if torch.cuda.is_available():
                f.write(f"最大GPU内存使用: {torch.cuda.max_memory_allocated() / 1024**3:.2f}GB\n")
            
            f.write("=" * 80 + "\n")
        
        return self.log_file

def get_system_info():
    """获取系统详细信息"""
    info = {
        "os": f"{platform.system()} {platform.version()}",
        "python": platform.python_version(),
        "cpu_cores": psutil.cpu_count(logical=False),
        "cpu_threads": psutil.cpu_count(),
        "memory_total": psutil.virtual_memory().total / 1024**3,
        "memory_available": psutil.virtual_memory().available / 1024**3,
    }
    
    if torch.cuda.is_available():
        info["gpu"] = torch.cuda.get_device_name(0)
        info["cuda_version"] = torch.version.cuda
        info["gpu_memory"] = torch.cuda.get_device_properties(0).total_memory / 1024**3
        
    return info

def print_debug_summary(monitor_log_path):
    """打印调试信息摘要"""
    if not os.path.exists(monitor_log_path):
        print("未找到调试日志文件")
        return
        
    print("\n" + "=" * 80)
    print("训练调试信息摘要".center(80))
    print("=" * 80)
    
    # 读取日志文件的最后20行
    try:
        with open(monitor_log_path, "r") as f:
            lines = f.readlines()
            
        if len(lines) > 20:
            print(f"(显示最后20行，完整日志见: {monitor_log_path})")
            for line in lines[-20:]:
                print(line.strip())
        else:
            for line in lines:
                print(line.strip())
    except:
        print(f"无法读取日志文件: {monitor_log_path}")
