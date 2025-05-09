"""
训练损失可视化工具
用于将训练损失数据可视化为曲线图
"""
import os
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import numpy as np
from datetime import datetime

def plot_loss_curves(csv_path, output_dir=None, window_size=5, show=False):
    """
    绘制训练损失曲线图
    
    Args:
        csv_path (str): 损失数据CSV文件路径
        output_dir (str): 输出目录，默认为CSV所在目录
        window_size (int): 移动平均窗口大小
        show (bool): 是否显示图表
        
    Returns:
        str: 保存的图表路径
    """
    # 确定输出目录
    if output_dir is None:
        output_dir = os.path.dirname(csv_path)
    os.makedirs(output_dir, exist_ok=True)
    
    # 读取CSV数据
    try:
        df = pd.read_csv(csv_path)
        print(f"成功加载了 {len(df)} 条训练记录")
    except Exception as e:
        print(f"读取CSV文件时出错: {str(e)}")
        return None
    
    # 确保数据中包含必要的列
    required_cols = ['Step', 'Total Loss', 'Instance Loss', 'Class Loss']
    if not all(col in df.columns for col in required_cols):
        print(f"CSV文件缺少必要的列: {required_cols}")
        return None
    
    # 计算移动平均
    df_smooth = df.copy()
    if window_size > 1:
        df_smooth['Total Loss'] = df['Total Loss'].rolling(window=window_size).mean()
        df_smooth['Instance Loss'] = df['Instance Loss'].rolling(window=window_size).mean()
        df_smooth['Class Loss'] = df['Class Loss'].rolling(window=window_size).mean()
        # 使用ffill()代替废弃的fillna(method='ffill')
        df_smooth = df_smooth.ffill()
    
    # 设置图表样式 - 使用兼容的样式
    try:
        # 检查可用样式
        available_styles = plt.style.available
        print(f"可用的样式: {available_styles}")
        
        # 尝试使用"dark_background"或默认样式
        if 'seaborn-v0_8-darkgrid' in available_styles:
            plt.style.use('seaborn-v0_8-darkgrid')
        elif 'dark_background' in available_styles:
            plt.style.use('dark_background')
        else:
            # 不设置样式，使用默认
            pass
    except Exception as e:
        print(f"设置样式时出错: {e}")
        print("继续使用默认样式...")
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # 绘制三种损失曲线
    ax.plot(df['Step'], df['Total Loss'], 'b-', alpha=0.3, label='总损失')
    ax.plot(df['Step'], df['Instance Loss'], 'g-', alpha=0.3, label='实例损失')
    ax.plot(df['Step'], df['Class Loss'], 'r-', alpha=0.3, label='类别损失')
    
    # 绘制平滑后的曲线
    ax.plot(df_smooth['Step'], df_smooth['Total Loss'], 'b-', linewidth=2, label=f'总损失 (平滑)')
    ax.plot(df_smooth['Step'], df_smooth['Instance Loss'], 'g-', linewidth=2, label=f'实例损失 (平滑)')
    ax.plot(df_smooth['Step'], df_smooth['Class Loss'], 'r-', linewidth=2, label=f'类别损失 (平滑)')
    
    # 添加标题和标签
    max_step = df['Step'].max()
    ax.set_title(f'DreamBooth 训练损失曲线 (步骤 0-{max_step})', fontsize=16)
    ax.set_xlabel('训练步骤', fontsize=14)
    ax.set_ylabel('损失值', fontsize=14)
    
    # 设置Y轴范围
    # 计算合理的y轴上限，排除极端值
    y_values = np.concatenate([df['Total Loss'].values, 
                              df['Instance Loss'].values, 
                              df['Class Loss'].values])
    # 移除极端值（超过95%分位数的3倍）
    y_95_percentile = np.percentile(y_values, 95)
    y_values = y_values[y_values < y_95_percentile * 3]
    y_max = max(y_values) * 1.1  # 给上限增加10%的空间
    ax.set_ylim(0, y_max)
    
    # 添加图例
    ax.legend(fontsize=12)
    
    # 添加网格
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # 添加当前训练进度标记
    progress = (max_step / 2000) * 100  # 假设总步骤为2000
    ax.text(0.02, 0.02, f'训练进度: {progress:.2f}%', transform=ax.transAxes, 
            fontsize=12, bbox=dict(facecolor='white', alpha=0.7))
    
    # 添加生成时间戳
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    plt.figtext(0.99, 0.01, f'生成时间: {timestamp}', 
                horizontalalignment='right', fontsize=8)
    
    # 保存图表
    timestamp_short = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = os.path.join(output_dir, f'training_loss_{timestamp_short}.png')
    plt.tight_layout()
    plt.savefig(output_file, dpi=120)
    print(f"图表已保存到: {output_file}")
    
    # 同时保存一个固定名称的版本（方便自动更新查看）
    fixed_output_file = os.path.join(output_dir, 'training_loss_latest.png')
    plt.savefig(fixed_output_file, dpi=120)
    
    # 显示图表（可选）
    if show:
        plt.show()
    else:
        plt.close()
    
    return output_file

def main():
    """命令行入口点"""
    parser = argparse.ArgumentParser(description='绘制DreamBooth训练损失曲线')
    parser.add_argument('--csv_path', type=str, required=True, 
                        help='训练损失CSV文件路径')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='图表输出目录，默认与CSV文件相同目录')
    parser.add_argument('--window_size', type=int, default=5,
                        help='移动平均窗口大小，用于曲线平滑')
    parser.add_argument('--show', action='store_true',
                        help='生成后显示图表')
    
    args = parser.parse_args()
    plot_loss_curves(args.csv_path, args.output_dir, args.window_size, args.show)

if __name__ == "__main__":
    main()
