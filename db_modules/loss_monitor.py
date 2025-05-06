"""
DreamBooth 损失监控模块
用于监控和优化训练过程中的损失，特别是解决实例损失为零的问题
"""
import os
import numpy as np
import torch
import logging
import time
from PIL import Image

logger = logging.getLogger(__name__)

class LossMonitor:
    """损失监控器，用于检测和修正训练问题"""
    
    def __init__(self, threshold=1e-5, patience=10):
        """
        初始化损失监控器
        
        Args:
            threshold: 损失阈值，低于此值视为异常
            patience: 容忍连续异常的次数
        """
        self.threshold = threshold
        self.patience = patience
        self.instance_loss_history = []
        self.prior_loss_history = []
        self.total_loss_history = []
        self.zero_loss_counter = 0
        self.warning_issued = False
        
    def check_loss(self, instance_loss, prior_loss=None, total_loss=None, step=0):
        """
        检查损失值并记录
        
        返回:
            bool: 如果损失值正常返回True，否则返回False
        """
        # 记录损失
        self.instance_loss_history.append(float(instance_loss))
        if prior_loss is not None:
            self.prior_loss_history.append(float(prior_loss))
        if total_loss is not None:
            self.total_loss_history.append(float(total_loss))
            
        # 检查实例损失是否接近零
        if abs(instance_loss) < self.threshold:
            self.zero_loss_counter += 1
            
            if self.zero_loss_counter >= self.patience and not self.warning_issued:
                logger.warning(f"检测到实例损失接近零 ({instance_loss:.8f})，已连续{self.zero_loss_counter}步")
                self.warning_issued = True
                return False
        else:
            self.zero_loss_counter = 0
            self.warning_issued = False
            
        return True
    
    def get_stats(self):
        """获取损失统计数据"""
        stats = {}
        
        if self.instance_loss_history:
            stats['instance_loss'] = {
                'current': self.instance_loss_history[-1],
                'mean': np.mean(self.instance_loss_history),
                'min': np.min(self.instance_loss_history),
                'max': np.max(self.instance_loss_history)
            }
            
        if self.prior_loss_history:
            stats['prior_loss'] = {
                'current': self.prior_loss_history[-1],
                'mean': np.mean(self.prior_loss_history),
                'min': np.min(self.prior_loss_history),
                'max': np.max(self.prior_loss_history)
            }
            
        if self.total_loss_history:
            stats['total_loss'] = {
                'current': self.total_loss_history[-1],
                'mean': np.mean(self.total_loss_history),
                'min': np.min(self.total_loss_history),
                'max': np.max(self.total_loss_history)
            }
            
        return stats
    
    def suggest_fixes(self):
        """根据损失模式提出修复建议"""
        suggestions = []
        
        if not self.instance_loss_history:
            return ["尚无损失数据，请等待训练开始"]
        
        # 检查实例损失为零的情况
        if self.zero_loss_counter > 0:
            suggestions.extend([
                "实例损失接近零，可能存在以下问题:",
                "0. 检查训练日志中 '[compute_loss DEBUG L1] Batch instance_count' 是否持续为0。如果是，表明批次中没有实例图像，请仔细检查您的数据集 (Dataset) 类和数据加载器 (DataLoader) 配置，特别是确保 'is_instance' 标志被正确设置为 True 对您的实例图像。",
                "1. 实例图像质量问题（如果 instance_count > 0 但损失仍为0，请检查图像是否清晰且多样化）。",
                "2. 学习率可能过低（如果 instance_count > 0）。",
                "3. 可能需要增加训练步数（如果 instance_count > 0）。",
                "4. 尝试设置更高的实例损失权重（如果 instance_count > 0）。"
            ])
            
        # 检查损失不稳定的情况
        loss_std = np.std(self.instance_loss_history[-20:]) if len(self.instance_loss_history) >= 20 else 0
        if loss_std > 0.1:
            suggestions.append("实例损失不稳定，建议降低学习率或增加批次大小")
            
        # 检查损失是否一直未下降
        if len(self.instance_loss_history) > 50:
            early_mean = np.mean(self.instance_loss_history[:20])
            recent_mean = np.mean(self.instance_loss_history[-20:])
            if recent_mean >= early_mean:
                suggestions.append("损失未见明显下降，可能需要调整模型超参数或检查数据集")
                
        return suggestions if suggestions else ["训练过程正常"]

def verify_instance_images(instance_dir):
    """
    验证实例图像是否有效
    
    Args:
        instance_dir: 实例图像目录
    
    返回:
        tuple: (有效图像数量, 问题列表)
    """
    if not os.path.exists(instance_dir):
        return 0, ["实例图像目录不存在"]
        
    issues = []
    valid_count = 0
    
    # 检查目录中的图像
    image_files = [f for f in os.listdir(instance_dir) 
                   if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))]
    
    if not image_files:
        return 0, ["未找到图像文件"]
    
    for img_file in image_files:
        try:
            img_path = os.path.join(instance_dir, img_file)
            with Image.open(img_path) as img:
                # 检查图像尺寸
                w, h = img.size
                if w < 256 or h < 256:
                    issues.append(f"图像 {img_file} 尺寸过小 ({w}x{h})")
                    continue
                    
                # 检查图像模式
                if img.mode not in ['RGB', 'RGBA']:
                    issues.append(f"图像 {img_file} 模式不支持 ({img.mode})")
                    continue
                
                # 检查图像是否全黑或全白
                if img.mode == 'RGB':
                    img_array = np.array(img)
                    if np.mean(img_array) < 10 or np.mean(img_array) > 245:
                        issues.append(f"图像 {img_file} 可能是全黑或全白图像")
                        continue
                
                valid_count += 1
                
        except Exception as e:
            issues.append(f"处理图像 {img_file} 时出错: {str(e)}")
    
    # 总结
    if valid_count == 0:
        issues.append("没有有效的实例图像，这将导致实例损失为零")
    elif valid_count < 5:
        issues.append(f"只有 {valid_count} 张有效图像，建议使用 5-10 张高质量图像")
    
    return valid_count, issues

def adjust_training_params(params, monitor=None):
    """
    根据损失情况调整训练参数
    
    Args:
        params: 当前训练参数
        monitor: 损失监控器对象
        
    返回:
        dict: 调整后的参数
    """
    adjusted = params.copy()
    
    # 如果没有监控器数据，设置默认优化参数
    if not monitor or not monitor.instance_loss_history:
        # 为防止实例损失为零，设置更高的学习率和较大的损失权重
        adjusted["learning_rate"] = max(1e-5, params.get("learning_rate", 5e-6))
        adjusted["instance_loss_weight"] = 1.0
        return adjusted
    
    # 有监控数据时根据损失情况调整
    stats = monitor.get_stats()
    
    # 如果实例损失接近零，增加学习率和损失权重
    if stats['instance_loss']['current'] < 1e-4:
        adjusted["learning_rate"] = min(2e-5, params.get("learning_rate", 5e-6) * 2)
        adjusted["instance_loss_weight"] = 1.5
        adjusted["train_text_encoder"] = True  # 确保训练文本编码器
        
    return adjusted
