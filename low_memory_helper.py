"""
低内存训练辅助模块 - 为显存有限的GPU优化DreamBooth训练
"""
import os
import torch
import gc
from contextlib import contextmanager

class LowMemoryConfig:
    """低内存训练配置"""
    def __init__(self, 
                 attention_slice_size=4,
                 train_text_encoder=False,
                 use_8bit_adam=True,
                 prior_generation_precision="fp16",
                 prior_images_small=10,
                 gradient_checkpointing=True,
                 max_train_steps_small=400,
                 offload_optimizer=False,
                 aggressive_gc=True):
        self.attention_slice_size = attention_slice_size  # 注意力计算切片大小
        self.train_text_encoder = train_text_encoder      # 是否训练文本编码器
        self.use_8bit_adam = use_8bit_adam                # 是否使用8位精度的Adam优化器
        self.prior_generation_precision = prior_generation_precision  # 先验图像生成精度
        self.prior_images_small = prior_images_small      # 小内存设备的先验图像数量
        self.gradient_checkpointing = gradient_checkpointing  # 是否使用梯度检查点
        self.max_train_steps_small = max_train_steps_small  # 小内存设备默认训练步数
        self.offload_optimizer = offload_optimizer        # 是否将优化器状态卸载到CPU
        self.aggressive_gc = aggressive_gc                # 是否频繁进行内存回收

    @classmethod
    def create_for_gpu(cls, gpu_memory_gb):
        """根据GPU内存大小创建适合的配置"""
        if gpu_memory_gb < 6:  # 极小内存 (4GB)
            return cls(
                attention_slice_size=1,
                train_text_encoder=False,
                use_8bit_adam=True,
                prior_generation_precision="fp16",
                prior_images_small=5,
                gradient_checkpointing=True,
                max_train_steps_small=300,
                offload_optimizer=True,
                aggressive_gc=True
            )
        elif gpu_memory_gb < 9:  # 小内存 (6-8GB)
            return cls(
                attention_slice_size=4, 
                train_text_encoder=False,
                use_8bit_adam=True,
                prior_generation_precision="fp16",
                prior_images_small=10,
                gradient_checkpointing=True
            )
        elif gpu_memory_gb < 13:  # 中等内存 (10-12GB)
            return cls(
                attention_slice_size=8,
                train_text_encoder=True,
                use_8bit_adam=True,
                prior_images_small=20
            )
        else:  # 大内存 (16GB+)
            return cls(
                attention_slice_size=0,
                train_text_encoder=True,
                use_8bit_adam=False,
                prior_images_small=50,
                gradient_checkpointing=False
            )

def optimize_for_inference(model, device="cuda"):
    """优化模型用于推理，减少内存占用"""
    # 启用融合kernels
    try:
        model.enable_xformers_memory_efficient_attention()
        print("已启用 xformers 优化")
    except (AttributeError, ImportError):
        try:
            model.enable_attention_slicing()
            print("已启用注意力切片")
        except:
            print("无法启用内存优化方法")
    
    # 使用半精度
    if "cuda" in device:
        try:
            import torch.nn.functional as F
            
            # 创建推理模型的半精度副本
            model = model.to(torch.float16)
            print("已将模型转换为半精度")
        except Exception as e:
            print(f"无法转换为半精度: {e}")
    
    return model

@contextmanager
def low_memory_context(config):
    """临时上下文，用于在低内存环境中执行操作"""
    if config.aggressive_gc:
        # 在执行前清理内存
        gc.collect()
        torch.cuda.empty_cache()
    
    try:
        # 临时修改PyTorch的内存分配行为
        old_fraction = torch.cuda.get_device_properties(0).total_memory
        torch.cuda.set_per_process_memory_fraction(0.9)  # 限制内存使用为90%
        
        yield  # 执行上下文中的代码
    
    finally:
        # 恢复原始设置并清理
        if config.aggressive_gc:
            gc.collect()
            torch.cuda.empty_cache()
        torch.cuda.set_per_process_memory_fraction(1.0)  # 恢复默认内存限制

def get_optimal_settings(gpu_memory_gb=None):
    """获取针对当前GPU最优的训练设置"""
    if gpu_memory_gb is None:
        if torch.cuda.is_available():
            gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        else:
            gpu_memory_gb = 0
    
    # 根据GPU内存返回推荐参数
    config = LowMemoryConfig.create_for_gpu(gpu_memory_gb)
    
    if gpu_memory_gb < 6:
        recommendation = "极小内存模式 (4GB): 仅训练U-Net，使用非常小的批次和较少的先验图像"
    elif gpu_memory_gb < 9:
        recommendation = "小内存模式 (8GB): 仅训练U-Net，启用渐变检查点和8位精度Adam优化器"
    elif gpu_memory_gb < 13:
        recommendation = "中等内存模式 (12GB): 可以训练文本编码器，使用更多的先验图像"
    else:
        recommendation = "大内存模式 (16GB+): 可以使用全精度和较大的批次"
    
    return config, recommendation

# 测试示例
if __name__ == "__main__":
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        config, recommendation = get_optimal_settings(gpu_memory)
        print(f"检测到GPU: {torch.cuda.get_device_name(0)}，内存: {gpu_memory:.2f}GB")
        print(f"推荐配置: {recommendation}")
        print(f"注意力切片大小: {config.attention_slice_size}")
        print(f"训练文本编码器: {config.train_text_encoder}")
        print(f"使用8位Adam: {config.use_8bit_adam}")
        print(f"先验图像数量: {config.prior_images_small}")
    else:
        print("未检测到GPU")
