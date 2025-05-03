# DreamBooth 安装指南

DreamBooth需要一些特定的依赖项才能正常运行。本指南将帮助您解决常见的安装问题。

## 基本安装步骤

### 1. 创建独立环境

推荐使用Conda创建一个独立的环境:

```bash
conda create -n dreambooth python=3.9
conda activate dreambooth
```

### 2. 安装PyTorch

根据您的CUDA版本选择合适的PyTorch安装命令:

**CUDA 11.8 (新的NVIDIA GPU)**:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**CUDA 11.7**:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117
```

**仅CPU或macOS**:
```bash
pip install torch torchvision torchaudio
```

### 3. 安装兼容的依赖项

最新版本可能会遇到兼容性问题，我们推荐使用这些兼容版本:

```bash
pip install diffusers==0.19.3 transformers==4.30.2 accelerate==0.21.0
pip install Pillow tqdm numpy
```

### 4. 可选: 安装额外的加速组件

如果您有NVIDIA GPU，可以安装xformers以加速训练:

```bash
pip install xformers
```

## 常见问题解决

### 导入错误 'No module named triton'

这通常与Flash Attention相关。解决方案:

1. 使用兼容的库版本:
   ```bash
   pip install diffusers==0.19.3 transformers==4.30.2 accelerate==0.21.0
   ```

2. 或者安装triton:
   ```bash
   pip install triton
   ```

### 内存不足错误

如果在训练时遇到内存不足错误:

1. 减少先验图像数量: `--prior_images 50` 
2. 禁用文本编码器训练: 移除 `--train_text_encoder` 选项
3. 使用较小的模型: `--small_model`

### 导入错误 'ImportError: cannot import name...'

库版本不兼容的常见问题。尝试安装这些兼容性更好的版本:

```bash
pip install diffusers==0.14.0 transformers==4.25.1 accelerate==0.15.0
```

## 验证安装

安装完成后，运行以下命令验证环境:

```bash
python -c "import torch; import diffusers; import transformers; print('GPU可用:' if torch.cuda.is_available() else 'GPU不可用:',torch.cuda.get_device_name(0) if torch.cuda.is_available() else ''); print('Diffusers版本:', diffusers.__version__); print('Transformers版本:', transformers.__version__)"
```

## 获取帮助

运行以下命令显示安装帮助:

```bash
python dreambooth_implementation.py --install_help
```
