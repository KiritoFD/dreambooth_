"""
模型工具模块 - 提供各种处理和管理模型的工具函数
"""

import torch
import logging
from typing import Dict, Any, Union, List, Optional, Tuple

logger = logging.getLogger(__name__)

def ensure_device_consistency(
    model_name: str, 
    model: torch.nn.Module, 
    input_tensors: Dict[str, torch.Tensor], 
    target_device: Optional[torch.device] = None
) -> Dict[str, torch.Tensor]:
    """
    确保输入张量与模型在同一设备上
    
    Args:
        model_name: 模型名称（用于日志记录）
        model: 模型实例
        input_tensors: 包含输入张量的字典
        target_device: 目标设备，如果未指定，则使用模型的设备
        
    Returns:
        字典，包含位于正确设备上的张量
    """
    # 确定模型的设备
    try:
        model_device = next(model.parameters()).device
    except StopIteration:
        # 模型没有参数，使用当前设备
        model_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 如果指定了目标设备，使用指定的设备
    device = target_device if target_device is not None else model_device
    
    # 将所有输入张量移动到正确的设备
    output_tensors = {}
    inconsistent_count = 0
    
    for name, tensor in input_tensors.items():
        if isinstance(tensor, torch.Tensor):
            # 如果张量在错误的设备上，移动它
            if tensor.device != device:
                inconsistent_count += 1
                output_tensors[name] = tensor.to(device)
            else:
                output_tensors[name] = tensor
        elif isinstance(tensor, dict):
            # 递归处理嵌套字典
            nested_dict = {}
            for k, v in tensor.items():
                if isinstance(v, torch.Tensor) and v.device != device:
                    inconsistent_count += 1
                    nested_dict[k] = v.to(device)
                else:
                    nested_dict[k] = v
            output_tensors[name] = nested_dict
        else:
            # 非张量值直接传递
            output_tensors[name] = tensor
    
    # 记录不一致的张量数量
    if inconsistent_count > 0:
        logger.warning(f"已修正 {model_name} 的 {inconsistent_count} 个输入张量到 {device} 设备")
    
    return output_tensors

def move_model_to_device(model: torch.nn.Module, device: torch.device) -> torch.nn.Module:
    """
    将模型移动到指定设备，并确保模型知道它在该设备上
    
    Args:
        model: 要移动的模型
        device: 目标设备
        
    Returns:
        位于指定设备上的模型
    """
    # 检查模型是否已经在目标设备上
    try:
        current_device = next(model.parameters()).device
        if current_device == device:
            return model
    except StopIteration:
        # 模型没有参数
        pass
    
    # 移动模型到目标设备
    model = model.to(device)
    
    # 对于某些模型，可能需要显式设置内部设备属性
    if hasattr(model, 'device') and isinstance(model.device, torch.device):
        model.device = device
    
    return model

def get_model_device_info(model_dict: Dict[str, torch.nn.Module]) -> Dict[str, str]:
    """
    获取多个模型的设备信息
    
    Args:
        model_dict: 包含模型的字典，键为模型名称，值为模型实例
        
    Returns:
        字典，包含每个模型的设备信息
    """
    device_info = {}
    
    for name, model in model_dict.items():
        try:
            device = next(model.parameters()).device
            device_info[name] = str(device)
        except StopIteration:
            device_info[name] = "未知（没有参数）"
        except Exception as e:
            device_info[name] = f"错误：{str(e)}"
    
    return device_info
