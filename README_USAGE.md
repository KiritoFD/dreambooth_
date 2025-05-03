# DreamBooth 实现指南

这个项目提供了一个基于论文《DreamBooth: Fine Tuning Text-to-Image Diffusion Models for Subject-Driven Generation》的实现。

## 系统要求

- Python 3.8+
- PyTorch 1.12+
- CUDA 支持的GPU (最少8GB显存，推荐16GB+)
- 必要的依赖（详见下文）

## 安装

1. 克隆本仓库:
```bash
git clone https://github.com/yourusername/dreambooth.git
cd dreambooth
```

2. 安装依赖:
```bash
pip install torch accelerate transformers diffusers==0.21.4 tqdm pillow
```

## 使用方法

### 准备数据

1. 创建一个文件夹用于存放您想要教模型学习的特定主题图像:
```bash
mkdir -p instance_images
```

2. 将3-5张图片放入此文件夹。图片应该:
   - 清晰呈现主题（如狗、猫、特定物品等）
   - 有不同的视角和姿势
   - 最好有简单的背景
   - 大小至少为512x512像素

### 训练模型

```bash
python dreambooth_implementation.py --train \
    --model_name "runwayml/stable-diffusion-v1-5" \
    --instance_data_dir "./instance_images" \
    --class_prompt "a dog" \
    --steps 1000 \
    --train_text_encoder
```

参数说明:
- `--model_name`: 使用的预训练模型
- `--instance_data_dir`: 包含您的主题图像的目录
- `--class_prompt`: 您主题的一般类别（如："a dog", "a cat", "a watch"）
- `--steps`: 训练步数
- `--train_text_encoder`: 设置此标志以同时微调文本编码器（推荐）

### 生成图像

```bash
python dreambooth_implementation.py --infer \
    --model_path "./output" \
    --prompt "a [identifier] dog on the moon" \
    --num_images 4
```

参数说明:
- `--model_path`: 保存微调模型的路径
- `--prompt`: 生成提示，使用[identifier]占位符或使用实际标识符
- `--num_images`: 要生成的图像数量

## 推荐的预训练模型

根据DreamBooth论文和实践经验，以下是推荐的预训练模型:

1. **runwayml/stable-diffusion-v1-5** - 最稳定的选择，与论文兼容性好
   - 适合大多数用例
   - 显存要求适中（约10GB）

2. **CompVis/stable-diffusion-v1-4** - 较早版本，也可使用
   - 与论文中使用的模型更接近
   - 显存要求较低（约8GB）

3. **stabilityai/stable-diffusion-2-1** - 更新版本
   - 生成质量更高
   - 显存要求略高（约12GB）

4. **stabilityai/stable-diffusion-xl-base-1.0** - SDXL模型
   - 提供最高质量和最佳细节
   - 显存要求高（约16GB+）

对于初次尝试，建议使用 **runwayml/stable-diffusion-v1-5**，它提供了良好的平衡。

## 提示和技巧

1. **数据质量**: 输入图像的质量直接影响结果
2. **训练时间**: 一般来说，1000步足够获得良好结果
3. **先验保留损失**: 默认权重为1.0，可以调整以平衡特定性和多样性
4. **推理提示**: 在提示中使用"标识符+类别名称"作为基础，加入其他描述

## 原理简介

DreamBooth通过几个关键机制工作:

1. **稀有标识符**: 使用罕见词汇将特定主题与普通类别区分开
2. **类别先验保留**: 通过生成类别示例来防止语言漂移
3. **微调策略**: 微调所有层以获得最佳保真度

详细原理请参考原论文和项目中的README.md文件。
