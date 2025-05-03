# DreamBooth 精简版实现

这是一个精简版的 DreamBooth 实现，基于论文《DreamBooth: Fine Tuning Text-to-Image Diffusion Models for Subject-Driven Generation》，
专注于核心功能，提供简洁的训练和推理接口。

## 文件结构

- `dreambooth_implementation.py` - 核心训练网络和流程
- `utils.py` - 辅助函数和内存管理
- `main.py` - 命令行接口

## 快速使用

### 训练模式

```bash
python main.py --train --instance_data_dir ./my_images --class_prompt "a dog"
```

可选参数:
- `--model_name` - 预训练模型名称（默认使用SD1.4）
- `--steps` - 训练步数（默认800）
- `--train_text_encoder` - 是否同时训练文本编码器
- `--prior_images` - 先验保留的类别图像数量（默认10）

### 推理模式

```bash
python main.py --infer --model_path ./output --prompt "a sks dog on the beach"
```

## 依赖项

```bash
pip install torch diffusers==0.19.3 transformers==4.30.2 accelerate tqdm pillow
```

## 论文引用

Ruiz, N., Li, Y., Jampani, V., Pritch, Y., Rubinstein, M., & Aberman, K. (2022).
DreamBooth: Fine Tuning Text-to-Image Diffusion Models for Subject-Driven Generation.
