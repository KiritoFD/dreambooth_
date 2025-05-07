# Stable Diffusion 模型下载与使用指南

本指南将帮助您正确下载和配置Stable Diffusion模型，以便在DreamBooth训练中使用。

## 方法一：使用Hugging Face CLI下载模型

1. **安装Hugging Face CLI工具**

   ```bash
   pip install huggingface_hub
   ```

2. **登录Hugging Face（首次使用需要）**

   ```bash
   huggingface-cli login
   ```
   
   按提示输入您的Hugging Face token（可以从[这里](https://huggingface.co/settings/tokens)获取）

3. **下载Stable Diffusion模型**

   ```bash
   huggingface-cli download --resume-download CompVis/stable-diffusion-v1-5 --local-dir ./models/stable-diffusion-v1-5
   ```

   这将把模型下载到`./models/stable-diffusion-v1-5`目录

4. **修改config.json中的模型路径**

   ```json
   "paths": {
     "pretrained_model_name_or_path": "./models/stable-diffusion-v1-5",
     ...
   }
   ```

## 方法二：使用Python脚本下载

创建一个`download_model.py`文件：

```python
from huggingface_hub import snapshot_download

# 下载模型
model_path = snapshot_download(
    repo_id="CompVis/stable-diffusion-v1-5",
    local_dir="./models/stable-diffusion-v1-5",
    resume_download=True
)
print(f"模型已下载到: {model_path}")
```

然后运行：

```bash
python download_model.py
```

## 方法三：直接使用在线模型

如果您有良好的网络连接，可以直接在`config.json`中使用在线模型标识符：

```json
"paths": {
  "pretrained_model_name_or_path": "CompVis/stable-diffusion-v1-5",
  ...
}
```

系统会自动从Hugging Face下载必要的文件。请确保您的网络能够访问huggingface.co。

## 使用现有下载的模型

如果您已经下载了模型（如您目录中的`models--stable-diffusion-v1-5--stable-diffusion-v1-5`），可以直接使用该路径：

```json
"paths": {
  "pretrained_model_name_or_path": "C:/GitHub/dreambooth/models--stable-diffusion-v1-5--stable-diffusion-v1-5/snapshots/451f4fe16113bff5a5d2269ed5ad43b0592e9a14",
  ...
}
```

## 常见问题排查

1. **无法下载模型**
   - 确认您的网络可以访问huggingface.co
   - 如果在中国大陆，可能需要使用代理

2. **模型下载后无法加载**
   - 确认模型目录结构完整，包含`tokenizer`、`text_encoder`、`unet`等子目录
   - 检查路径是否正确，避免使用特殊字符

3. **权限问题**
   - 某些模型(如SD-2.1)需要接受使用条款，请先在浏览器中访问模型页面并接受条款

4. **存储空间不足**
   - Stable Diffusion模型约需4-7GB存储空间，请确保有足够空间

## 其他流行的Stable Diffusion模型变体

- **Stable Diffusion 2.1**: `stabilityai/stable-diffusion-2-1`
- **Stable Diffusion XL**: `stabilityai/stable-diffusion-xl-base-1.0`
- **Anything V3**: `Linaqruf/anything-v3.0`

使用这些模型变体时，只需在`config.json`中替换相应的模型ID或路径即可。
