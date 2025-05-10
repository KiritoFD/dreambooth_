"""
DreamBooth 内存优化模块
提供针对不同GPU内存大小的优化策略
"""
import os
import gc
import torch
import contextlib


class MemoryConfig:
    """内存优化配置"""
    def __init__(self):
        self.attention_slice_size = 0
        self.gradient_checkpointing = False
        self.use_8bit_adam = False
        self.disable_text_encoder_training = False
        self.inference_steps = 50
        self.xformers_optimization = True
        self.offload_to_cpu = False
        self.prior_images = 100
        self.aggressive_gc = False


def get_optimal_settings(gpu_memory_gb=None):
    """
    根据GPU内存大小获取最佳训练设置
    
    Args:
        gpu_memory_gb (float): GPU内存大小(GB)
        
    Returns:
        dict: 优化配置字典
    """
    if gpu_memory_gb is None:
        if torch.cuda.is_available():
            gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        else:
            gpu_memory_gb = 0
    
    config = MemoryConfig()
    
    # 根据GPU内存大小进行配置
    if gpu_memory_gb < 6:  # 极低内存 (4-6GB)
        config.attention_slice_size = 1
        config.gradient_checkpointing = True
        config.use_8bit_adam = True
        config.disable_text_encoder_training = True
        config.inference_steps = 30
        config.xformers_optimization = True
        config.offload_to_cpu = True
        config.prior_images = 5
        config.aggressive_gc = True
        
    elif gpu_memory_gb < 8:  # 低内存 (6-8GB)
        config.attention_slice_size = 2
        config.gradient_checkpointing = True
        config.use_8bit_adam = True
        config.disable_text_encoder_training = True
        config.inference_steps = 40
        config.xformers_optimization = True
        config.prior_images = 10
        config.aggressive_gc = True
        
    elif gpu_memory_gb < 12:  # 中等内存 (8-12GB)
        config.attention_slice_size = 4
        config.gradient_checkpointing = True
        config.use_8bit_adam = True
        config.disable_text_encoder_training = False
        config.prior_images = 100000
        
    else:  # 高内存 (>12GB)
        # 默认设置，不需要特别优化
        config.prior_images = 50
    
    # 返回配置字典
    return {
        "attention_slice_size": config.attention_slice_size,
        "gradient_checkpointing": config.gradient_checkpointing,
        "use_8bit_adam": config.use_8bit_adam,
        "disable_text_encoder_training": config.disable_text_encoder_training,
        "inference_steps": config.inference_steps,
        "xformers_optimization": config.xformers_optimization,
        "offload_to_cpu": config.offload_to_cpu,
        "prior_images": config.prior_images,
        "aggressive_gc": config.aggressive_gc
    }


def optimize_model_for_training(unet, text_encoder, config):
    """
    根据配置优化模型以减少内存使用
    
    Args:
        unet: U-Net模型
        text_encoder: 文本编码器模型
        config: 优化配置
    """
    # 应用注意力切片
    if config["attention_slice_size"] > 0:
        try:
            if hasattr(unet, "set_attention_slice"):
                unet.set_attention_slice(config["attention_slice_size"])
                print(f"已设置注意力切片大小为 {config['attention_slice_size']}")
            else:
                unet.config.attention_slice_size = config["attention_slice_size"]
                print(f"已设置注意力切片大小配置")
        except Exception as e:
            print(f"设置注意力切片失败: {e}")
    
    # 应用梯度检查点
    if config["gradient_checkpointing"]:
        try:
            unet.enable_gradient_checkpointing()
            if not config["disable_text_encoder_training"]:
                text_encoder.gradient_checkpointing_enable()
            print("已启用梯度检查点以减少内存使用")
        except Exception as e:
            print(f"启用梯度检查点失败: {e}")
    
    # 尝试应用xformers优化
    if config["xformers_optimization"]:
        try:
            import xformers
            unet.enable_xformers_memory_efficient_attention()
            print("已启用xformers内存优化")
        except ImportError:
            print("xformers未安装，跳过此优化")
        except Exception as e:
            print(f"启用xformers优化失败: {e}")
    
    return unet, text_encoder


def optimize_for_inference(model, device="cuda:0"):
    """
    优化模型用于推理，减少内存占用
    
    Args:
        model: 扩散模型管道
        device: 推理设备
    
    Returns:
        优化后的模型
    """
    # 设置为最节约内存的方式
    if device == "cuda:0" and torch.cuda.is_available():
        # 激活GPU内存优化
        try:
            # 使用torch 2.0的编译功能
            if hasattr(torch, "compile") and torch.__version__ >= "2":
                model.unet = torch.compile(model.unet, mode="reduce-overhead")
                print("已使用torch.compile优化UNet")
        except Exception as e:
            print(f"编译优化失败: {e}")
        
        # 尝试启用xformers
        try:
            import xformers
            model.enable_xformers_memory_efficient_attention()
            print("已启用xformers优化")
        except ImportError:
            try:
                model.enable_attention_slicing(slice_size=1)
                print("已启用注意力切片")
            except:
                print("无法启用注意力优化")
        
        # 转换为半精度
        model.to(torch.float16)
        print("已将模型转换为半精度以节省内存")
        
        # VAE优化
        try:
            from diffusers.models import AutoencoderKL
            if isinstance(model.vae, AutoencoderKL):
                # 使用tiled VAE优化以处理大图像
                model.vae.enable_tiling()
                print("已启用VAE平铺以支持大图像")
        except:
            pass
    
    return model


@contextlib.contextmanager
def track_memory_usage(name="Memory usage"):
    """跟踪操作的内存使用"""
    if torch.cuda.is_available():
        start_memory = torch.cuda.memory_allocated() / 1024**3
        torch.cuda.reset_peak_memory_stats()
        yield
        end_memory = torch.cuda.memory_allocated() / 1024**3
        peak_memory = torch.cuda.max_memory_allocated() / 1024**3
        print(f"{name}: 使用 {end_memory:.2f}GB (峰值 {peak_memory:.2f}GB, 增加 {end_memory - start_memory:.2f}GB)")
    else:
        yield


def aggressive_memory_cleanup():
    """执行彻底的内存清理"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        # 尝试使用CUDA图形API释放更多内存
        try:
            torch._C._cuda_clearCublasWorkspaces()
        except:
            pass


def get_memory_status():
    """获取当前内存状态报告"""
    result = {
        "available_system_memory": 0,
        "used_system_memory": 0,
        "gpu_available": False,
        "gpu_total": 0,
        "gpu_used": 0,
        "gpu_free": 0
    }
    
    # 系统内存
    try:
        import psutil
        vm = psutil.virtual_memory()
        result["available_system_memory"] = vm.available / 1024**3
        result["used_system_memory"] = vm.used / 1024**3
    except:
        pass
    
    # GPU内存
    if torch.cuda.is_available():
        result["gpu_available"] = True
        result["gpu_total"] = torch.cuda.get_device_properties(0).total_memory / 1024**3
        result["gpu_used"] = torch.cuda.memory_allocated() / 1024**3
        result["gpu_free"] = result["gpu_total"] - result["gpu_used"]
    
    return result


def print_memory_stats():
    """打印当前内存统计信息"""
    stats = get_memory_status()
    
    print("\n" + "="*60)
    print("内存使用情况")
    print("="*60)
    
    print(f"系统内存: 已用 {stats['used_system_memory']:.2f}GB, 可用 {stats['available_system_memory']:.2f}GB")
    
    if stats["gpu_available"]:
        print(f"GPU内存: 已用 {stats['gpu_used']:.2f}GB / 总计 {stats['gpu_total']:.2f}GB "
              f"({stats['gpu_used']/stats['gpu_total']*100:.1f}%)")
    else:
        print("GPU未可用")
    
    print("="*60)
    

if __name__ == "__main__":
    # 简单测试
    print("内存优化工具测试")
    
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        config = get_optimal_settings(gpu_memory)
        
        print(f"\nGPU: {torch.cuda.get_device_name(0)} ({gpu_memory:.1f}GB)")
        print("\n推荐的内存优化配置:")
        for key, value in config.items():
            print(f"- {key}: {value}")
        
        print_memory_stats()
    else:
        print("未检测到GPU")
