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
        self.prior_images = 200
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
        config.prior_images = 200
        
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


def optimize_for_inference(model, device="cuda"):
    """
    优化模型用于推理，减少内存占用
    
    Args:
        model: 扩散模型管道
        device: 推理设备
    
    Returns:
        优化后的模型
    """
    # 设置为最节约内存的方式
    if device == "cuda" and torch.cuda.is_available():
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
        
        # 尝试重置CUDA设备，更激进的方式（谨慎使用）
        try:
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.reset_accumulated_memory_stats()
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


def enforce_memory_limits(gpu_memory_fraction=0.9):
    """
    强制执行内存限制，防止内存溢出
    
    Args:
        gpu_memory_fraction: GPU内存使用限制比例(0.0-1.0)
    """
    if torch.cuda.is_available():
        # 设置CUDA内存分配器限制
        torch.cuda.set_per_process_memory_fraction(gpu_memory_fraction)
        print(f"已限制GPU内存使用为总内存的{gpu_memory_fraction*100:.1f}%")
        
        # 尝试设置内存分配策略
        try:
            # 如果pytorch版本支持，设置更保守的内存分配策略
            if hasattr(torch.cuda, 'memory_stats') and callable(getattr(torch.cuda, 'memory_stats', None)):
                torch.cuda.empty_cache()
                print("设置保守内存分配策略")
        except:
            pass


def tensor_to_cpu_or_detach(tensor):
    """
    优化张量，减少内存使用
    将不需要梯度的张量转移到CPU或进行分离
    
    Args:
        tensor: 输入张量
    
    Returns:
        优化后的张量
    """
    if tensor is None:
        return None
        
    if isinstance(tensor, torch.Tensor):
        if tensor.requires_grad:
            return tensor.detach()
        else:
            return tensor.cpu() if tensor.device.type == "cuda" else tensor
    return tensor


def deep_tensor_cleanup(var_dict):
    """
    深度清理字典中的张量
    
    Args:
        var_dict: 包含张量的字典
    """
    for key, value in var_dict.items():
        if isinstance(value, torch.Tensor):
            del value
        elif isinstance(value, dict):
            deep_tensor_cleanup(value)
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, torch.Tensor):
                    del item
    
    # 显式触发垃圾回收
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def cyclic_memory_cleanup(frequency=10, current_step=0):
    """
    周期性执行内存清理
    每隔指定步数执行一次彻底清理
    
    Args:
        frequency: 清理频率（步数）
        current_step: 当前步数
    
    Returns:
        bool: 是否执行了清理
    """
    if current_step % frequency == 0:
        aggressive_memory_cleanup()
        return True
    return False


def optimize_sdxl_memory(unet, vae, text_encoder, text_encoder_2=None):
    """
    特别针对SDXL模型的内存优化
    
    Args:
        unet: UNet模型
        vae: VAE模型
        text_encoder: 第一文本编码器
        text_encoder_2: 第二文本编码器
    """
    # 使用half精度
    try:
        vae.half()
        print("已将VAE转换为半精度")
    except Exception as e:
        print(f"VAE转换为半精度失败: {e}")
    
    # 设置注意力处理器的cache大小
    try:
        if hasattr(unet, "set_attention_cake_size"):
            unet.set_attention_cache_size(1)  # 最小缓存大小
            print("已设置UNet注意力缓存大小为最小值")
    except:
        pass
    
    # 确保文本编码器中的权重按需加载
    if text_encoder is not None:
        if hasattr(text_encoder, "gradient_checkpointing_enable"):
            text_encoder.gradient_checkpointing_enable()
    
    if text_encoder_2 is not None:
        if hasattr(text_encoder_2, "gradient_checkpointing_enable"):
            text_encoder_2.gradient_checkpointing_enable()
    
    print("已应用SDXL特定内存优化")
    
    return unet, vae, text_encoder, text_encoder_2


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
