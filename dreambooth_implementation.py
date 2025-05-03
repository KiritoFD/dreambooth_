import os
import random
import numpy as np
import argparse
from pathlib import Path
from tqdm.auto import tqdm
from PIL import Image

# 首先检查依赖项
def check_dependencies():
    missing_deps = []
    try:
        import torch
    except ImportError:
        missing_deps.append("torch")
    
    try:
        import accelerate
    except ImportError:
        missing_deps.append("accelerate")
        
    try:
        import transformers
    except ImportError:
        missing_deps.append("transformers")
        
    try:
        import diffusers
    except ImportError:
        missing_deps.append("diffusers")
    
    return missing_deps

missing = check_dependencies()
if missing:
    print("缺少必要的依赖项，请先安装：")
    print(f"pip install {' '.join(missing)}")
    print("\n如果遇到Flash Attention相关错误，请尝试：")
    print("pip install diffusers==0.19.3 transformers==4.30.2 accelerate xformers")
    exit(1)

# 正常导入，现在有更好的错误处理
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

try:
    # 尝试导入主要模块
    from diffusers import StableDiffusionPipeline, DDPMScheduler, UNet2DConditionModel, AutoencoderKL
    from transformers import CLIPTextModel, CLIPTokenizer
except ImportError as e:
    print(f"导入错误: {e}")
    print("\n请尝试安装兼容版本的依赖项:")
    print("pip install diffusers==0.19.3 transformers==4.30.2 accelerate")
    print("如果您使用NVIDIA GPU，可以添加: xformers")
    exit(1)

from accelerate import Accelerator

class DreamBoothDataset(Dataset):
    def __init__(self, instance_images_path, class_images_path, tokenizer, size=512, center_crop=True):
        self.size = size
        self.center_crop = center_crop
        self.tokenizer = tokenizer
        
        # 简化图像加载
        self.instance_images = self._load_images(instance_images_path) if os.path.exists(instance_images_path) else []
        self.class_images = self._load_images(class_images_path) if class_images_path and os.path.exists(class_images_path) else []
    
    def _load_images(self, path):
        images = []
        if not path:
            return images
            
        for file in os.listdir(path):
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                try:
                    img = Image.open(os.path.join(path, file)).convert('RGB')
                    # 简化图像处理
                    if self.center_crop:
                        w, h = img.size
                        min_dim = min(w, h)
                        img = img.crop(((w-min_dim)//2, (h-min_dim)//2, (w+min_dim)//2, (h+min_dim)//2))
                    img = img.resize((self.size, self.size))
                    images.append(img)
                except Exception:
                    pass  # 静默跳过有问题的图像
        return images
    
    def __len__(self):
        return len(self.instance_images) + len(self.class_images)
    
    def __getitem__(self, idx):
        # 区分实例图像和类图像
        if idx < len(self.instance_images):
            image = self.instance_images[idx]
            is_instance = True
        else:
            image = self.class_images[idx - len(self.instance_images)]
            is_instance = False
        
        # 转换为torch张量
        image = torch.from_numpy(np.array(image).astype(np.float32) / 127.5 - 1.0)
        image = image.permute(2, 0, 1)
        
        return {"pixel_values": image, "is_instance": is_instance}

def find_rare_token(tokenizer, token_range=(5000, 10000)):
    """
    按照论文所述，查找稀有令牌作为标识符
    对于Stable Diffusion，我们使用CLIP tokenizer
    """
    token_id = random.randint(token_range[0], 10000)
    # 确保选择的是符合条件的token（3个或更少Unicode字符）
    token_text = tokenizer.decode([token_id]).strip()
    while len(token_text) > 3 or ' ' in token_text:
        token_id = random.randint(token_range[0], token_range[1])
        token_text = tokenizer.decode([token_id]).strip()
        
    return token_text

def generate_class_images(model, class_prompt, output_dir, num_samples=200):
    """
    生成类别图像以用于先验保留损失
    这是论文中提到的先验保留机制的关键部分
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"正在生成 {num_samples} 张类别图像用于先验保留...")
    
    # 将模型设置为推理模式
    model.safety_checker = None  # 禁用安全检查器以加快生成
    
    # 批量生成图像以加快速度
    batch_size = 4
    num_batches = (num_samples + batch_size - 1) // batch_size
    
    for batch_idx in tqdm(range(num_batches)):
        batch_prompts = [class_prompt] * min(batch_size, num_samples - batch_idx * batch_size)
        with torch.no_grad():
            outputs = model(batch_prompts, num_inference_steps=50, guidance_scale=7.5)
            
        for i, image in enumerate(outputs.images):
            img_idx = batch_idx * batch_size + i
            image.save(os.path.join(output_dir, f"class_{img_idx:04d}.png"))
            
    return output_dir

def download_small_model():
    """
    自动下载小型模型，适合低资源设备
    返回模型的路径或名称
    """
    print("选择适合低资源设备的小型模型...")
    
    # 定义小型模型选项 - 更新为较旧但稳定且兼容性更好的版本
    small_models = [
        "CompVis/stable-diffusion-v1-4",                # 较旧但稳定的SD1.4，兼容性好
        "runwayml/stable-diffusion-v1-5",               # 标准SD1.5
        "stabilityai/stable-diffusion-2-base",          # SD2基础版
    ]
    
    # 选择默认小型模型
    chosen_model = small_models[0]
    
    print(f"已选择模型: {chosen_model}")
    print(f"此模型与较旧版本的diffusers兼容性更好")
    print("模型将在首次使用时自动从Hugging Face下载")
    
    return chosen_model

# 添加这个函数来显示简洁的使用说明
def show_quick_help():
    """显示简洁的使用说明"""
    print("\n需要指定操作模式! 请使用以下参数之一:")
    print("  --train          训练模式")
    print("  --infer          推理模式")
    print("  --install_help   显示详细安装指南")
    print("\n基础示例:")
    print("  # 训练模式")
    print("  python dreambooth_implementation.py --train --instance_data_dir ./my_images --class_prompt \"a cat\"")
    print("  # 推理模式")
    print("  python dreambooth_implementation.py --infer --model_path ./output --prompt \"a sks cat on the beach\"")
    print("\n输入 ? 或 --help 查看完整参数列表")

def dreambooth_training(
    pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5",
    instance_data_dir="./instance_images",
    output_dir="./output",
    class_prompt="a dog",
    instance_prompt=None,  # 将由稀有令牌和类别名称组成
    learning_rate=5e-6,
    max_train_steps=1000,
    prior_preservation_weight=1.0,  # λ参数，控制先验保留损失的权重
    prior_generation_samples=200,
    gradient_accumulation_steps=1,
    train_text_encoder=True,  # 是否微调文本编码器
    train_batch_size=1,
    seed=42,
):
    # 设置种子以保证结果可复现
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    
    # 初始化加速器
    accelerator = Accelerator(
        gradient_accumulation_steps=gradient_accumulation_steps,
        mixed_precision="fp16"  # 使用16位混合精度训练以节省内存
    )
    
    # 初始化tokenizer
    tokenizer = CLIPTokenizer.from_pretrained(
        pretrained_model_name_or_path, 
        subfolder="tokenizer"
    )
    
    # 查找稀有令牌作为标识符，如论文3.2节所述
    identifier = find_rare_token(tokenizer)
    print(f"选中的稀有令牌标识符: '{identifier}'")
    
    # 构建实例提示词，遵循论文中"a [identifier] [class noun]"的格式
    if instance_prompt is None:
        class_name = class_prompt.replace("a ", "").strip()
        instance_prompt = f"a {identifier} {class_name}"
    
    print(f"实例提示词: '{instance_prompt}'")
    print(f"类别提示词: '{class_prompt}'")
    
    # 加载预训练模型
    print(f"加载预训练模型: {pretrained_model_name_or_path}")
    # 加载文本编码器
    text_encoder = CLIPTextModel.from_pretrained(
        pretrained_model_name_or_path, 
        subfolder="text_encoder"
    )
    
    # 加载 VAE
    vae = AutoencoderKL.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="vae"
    )

    # 加载U-Net
    unet = UNet2DConditionModel.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="unet",
    )
    
    # 加载用于生成类别图像的完整pipeline
    pipeline = StableDiffusionPipeline.from_pretrained(
        pretrained_model_name_or_path,
        vae=vae, # 确保pipeline使用相同的VAE
        text_encoder=text_encoder,
        unet=unet,
    )
    pipeline.to(accelerator.device)

    # 将 VAE 设置为评估模式并移至 GPU，通常不训练 VAE
    vae.requires_grad_(False)
    vae.to(accelerator.device, dtype=torch.float32) # VAE 通常保持在 fp32

    # 生成类别图像用于先验保留
    class_images_dir = os.path.join(output_dir, "class_images")
    if not os.path.exists(class_images_dir) or len(os.listdir(class_images_dir)) < prior_generation_samples:
        generate_class_images(pipeline, class_prompt, class_images_dir, prior_generation_samples)
    
    # 加载数据集
    dataset = DreamBoothDataset(
        instance_images_path=instance_data_dir,
        class_images_path=class_images_dir,
        tokenizer=tokenizer,
    )
    dataloader = DataLoader(dataset, batch_size=train_batch_size, shuffle=True)
    
    # 准备优化器
    # 如论文所述，最佳的主体保真度是通过微调所有层获得的
    if train_text_encoder:
        params_to_optimize = list(unet.parameters()) + list(text_encoder.parameters())
        print("将同时微调U-Net和文本编码器")
    else:
        params_to_optimize = unet.parameters()
        print("仅微调U-Net")
    
    optimizer = torch.optim.AdamW(
        params_to_optimize,
        lr=learning_rate,
        betas=(0.9, 0.999),
        weight_decay=1e-2,
    )
    
    # 准备噪声调度器
    noise_scheduler = DDPMScheduler.from_pretrained(
        pretrained_model_name_or_path, 
        subfolder="scheduler"
    )
    
    # 准备文本嵌入
    instance_text_inputs = tokenizer(
        instance_prompt,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt"
    )
    class_text_inputs = tokenizer(
        class_prompt,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt"
    )
    
    # 将模型、优化器和数据加载器准备用于加速训练
    # 注意：VAE 通常不需要 accelerator.prepare，因为它不参与梯度计算且保持 fp32
    unet, text_encoder, optimizer, dataloader = accelerator.prepare(
        unet, text_encoder, optimizer, dataloader
    )
    
    # 训练循环
    print("开始训练...")
    progress_bar = tqdm(range(max_train_steps), desc="训练进度", disable=not accelerator.is_main_process) # 仅在主进程显示进度条
    global_step = 0
    
    # VAE 缩放因子，从 Stable Diffusion 配置中获取
    vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)

    for epoch in range(1):  # 通常一个epoch就足够了
        unet.train()
        if train_text_encoder:
            text_encoder.train()
        else:
            text_encoder.eval()
        
        for step, batch in enumerate(dataloader):
            if global_step >= max_train_steps:
                break
                
            with accelerator.accumulate(unet):
                # 准备输入
                pixel_values = batch["pixel_values"].to(accelerator.device, dtype=vae.dtype) # 确保dtype匹配VAE输入
                is_instance = batch["is_instance"]

                # 使用 VAE 将图像编码为潜在表示
                with torch.no_grad():
                    # 将 pixel_values 移到 VAE 所在的设备和类型
                    latents = vae.encode(pixel_values).latent_dist.sample()
                    latents = latents * vae.config.scaling_factor # 使用配置中的缩放因子

                # 获取相应的文本嵌入
                with torch.no_grad():
                    # 获取 text_encoder 的实际设备
                    text_encoder_device = text_encoder.device
                    if train_text_encoder and torch.sum(is_instance).item() > 0:
                        encoder_hidden_states_instance = text_encoder(
                            instance_text_inputs.input_ids.to(text_encoder_device)
                        )[0]
                    else:
                        encoder_hidden_states_instance = None
                        
                    if train_text_encoder and torch.sum(~is_instance).item() > 0:
                        encoder_hidden_states_class = text_encoder(
                            class_text_inputs.input_ids.to(text_encoder_device)
                        )[0]
                    else:
                        encoder_hidden_states_class = None

                # 为潜在表示添加噪声
                noise = torch.randn_like(latents)
                timesteps = torch.randint(
                    0, 
                    noise_scheduler.config.num_train_timesteps,
                    (latents.shape[0],), 
                    device=latents.device
                ).long()
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
                
                # 预测噪声残差 - 更新autocast以避免警告
                with torch.amp.autocast('cuda' if torch.cuda.is_available() else 'cpu'):
                    instance_loss = 0.0
                    class_loss = 0.0
                    
                    # 处理实例样本
                    if torch.sum(is_instance).item() > 0:
                        # 确保encoder_hidden_states不为None
                        if encoder_hidden_states_instance is None:
                            with torch.no_grad():
                                encoder_hidden_states_instance = text_encoder(
                                    instance_text_inputs.input_ids.to(text_encoder_device)
                                )[0]
                        
                        # 确保 encoder_hidden_states 在与 noisy_latents 相同的设备上
                        encoder_hidden_states_instance = encoder_hidden_states_instance.to(noisy_latents.device)

                        instance_batch = {
                            "sample": noisy_latents[is_instance], # 使用 noisy_latents
                            "timestep": timesteps[is_instance],
                            "encoder_hidden_states": encoder_hidden_states_instance.repeat(torch.sum(is_instance).item(), 1, 1),
                        }
                        noise_pred_instance = unet(**instance_batch).sample
                        # 目标是原始噪声
                        instance_loss = F.mse_loss(noise_pred_instance.float(), noise[is_instance].float(), reduction="mean")
                    
                    # 处理类别样本（先验保留）
                    if torch.sum(~is_instance).item() > 0:
                        # 确保encoder_hidden_states不为None
                        if encoder_hidden_states_class is None:
                            with torch.no_grad():
                                encoder_hidden_states_class = text_encoder(
                                    class_text_inputs.input_ids.to(text_encoder_device)
                                )[0]

                        # 确保 encoder_hidden_states 在与 noisy_latents 相同的设备上
                        encoder_hidden_states_class = encoder_hidden_states_class.to(noisy_latents.device)

                        class_batch = {
                            "sample": noisy_latents[~is_instance], # 使用 noisy_latents
                            "timestep": timesteps[~is_instance],
                            "encoder_hidden_states": encoder_hidden_states_class.repeat(
                                torch.sum(~is_instance).item(), 1, 1
                            ),
                        }
                        
                        noise_pred_class = unet(**class_batch).sample
                        # 目标是原始噪声
                        class_loss = F.mse_loss(noise_pred_class.float(), noise[~is_instance].float(), reduction="mean")
                    
                    # 组合损失，应用先验保留权重
                    loss = instance_loss + prior_preservation_weight * class_loss
                
                # 反向传播
                accelerator.backward(loss)

                # 梯度裁剪 (可选但推荐)
                if accelerator.sync_gradients:
                    params_to_clip = (
                        list(unet.parameters()) + list(text_encoder.parameters()
                        if train_text_encoder
                        else unet.parameters())
                    )
                    accelerator.clip_grad_norm_(params_to_clip, 1.0) # 使用 1.0 作为最大范数

                # 优化步骤
                optimizer.step()
                optimizer.zero_grad()
                
            # 记录和打印进度
            if accelerator.is_main_process: # 仅在主进程更新和打印
                progress_bar.set_postfix({
                    "loss": loss.detach().item(),
                    "instance_loss": instance_loss.detach().item() if isinstance(instance_loss, torch.Tensor) else instance_loss,
                    "class_loss": class_loss.detach().item() if isinstance(class_loss, torch.Tensor) else class_loss
                })
                progress_bar.update(1)

            global_step += 1
            
            if global_step >= max_train_steps: # 在内部循环检查以立即停止
                break
        
        if global_step >= max_train_steps: # 如果是因为达到步数而退出外层循环
            break

    # 显式关闭进度条
    progress_bar.close()

    # 等待所有进程完成
    accelerator.wait_for_everyone()
    
    # 保存微调后的模型
    if accelerator.is_main_process: # 仅在主进程保存
        # 需要先解包装模型
        unet = accelerator.unwrap_model(unet)
        # text_encoder 只有在训练时才需要解包
        final_text_encoder = None
        if train_text_encoder:
            final_text_encoder = accelerator.unwrap_model(text_encoder)
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 从原始pipeline创建新的pipeline
        # 如果文本编码器没有被训练，我们需要从原始路径加载它以进行保存
        if not train_text_encoder:
            # 从原始路径加载未训练的文本编码器
            final_text_encoder = CLIPTextModel.from_pretrained(
                pretrained_model_name_or_path, subfolder="text_encoder"
            )

        # 确保加载原始 VAE，因为它没有被训练
        # 注意：VAE 已经在函数开始时加载，并且没有被修改，所以可以直接使用原始路径加载
        pipeline = StableDiffusionPipeline.from_pretrained(
            pretrained_model_name_or_path, # 从原始路径加载 VAE 和其他未训练组件（如 scheduler, safety_checker）
            unet=unet,                     # 使用训练后的 unet
            text_encoder=final_text_encoder, # 使用训练后或原始的 text_encoder
            # VAE 会自动从 pretrained_model_name_or_path 加载
        )
        
        # 保存微调后的模型
        pipeline.save_pretrained(output_dir)
        print(f"模型已保存到 {output_dir}")
        
        # 保存标识符以供将来推理使用
        with open(os.path.join(output_dir, "identifier.txt"), "w") as f:
            f.write(identifier)
    
    return None, identifier

def inference(
    model_path="./output",
    prompt=None,
    class_prompt="a dog",
    identifier=None,
    output_image_path="./generated_image.png",
    num_images=1,
    guidance_scale=7.5,
    num_inference_steps=50,
):
    # 确保目录存在
    if not os.path.exists(model_path):
        print(f"错误: 模型路径不存在: {model_path}")
        print("请确保您已经训练了模型或提供了正确的路径")
        return None

    # 改进的设备选择
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"正在加载模型到设备: {device}")
    
    try:
        # 加载微调后的模型
        pipeline = StableDiffusionPipeline.from_pretrained(model_path)
        pipeline = pipeline.to(device)
        
        # 如果有CUDA，尝试启用内存优化
        if device == "cuda":
            try:
                # 尝试启用内存优化
                pipeline.enable_attention_slicing()
                # 尝试检测并启用xFormers优化
                if hasattr(pipeline, "enable_xformers_memory_efficient_attention"):
                    pipeline.enable_xformers_memory_efficient_attention()
                    print("已启用xFormers优化以提高性能")
            except Exception as e:
                print(f"注意: 无法启用某些GPU优化: {e}")
    
        # 如果未提供标识符但存在标识符文件，则读取它
        if identifier is None and os.path.exists(os.path.join(model_path, "identifier.txt")):
            with open(os.path.join(model_path, "identifier.txt"), "r") as f:
                identifier = f.read().strip()
                print(f"从文件加载标识符: {identifier}")
        
        # 如果未提供提示词，则使用标识符创建
        if prompt is None and identifier is not None:
            class_name = class_prompt.replace("a ", "").strip()
            prompt = f"a {identifier} {class_name} wearing a hat"
        elif prompt is None:
            raise ValueError("需要提供prompt或identifier")
        
        # 生成图像
        print(f"使用提示词生成图像: '{prompt}'")
        outputs = pipeline(
            prompt,
            num_images_per_prompt=num_images,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps
        )
        
        # 保存所有生成的图像
        os.makedirs(os.path.dirname(output_image_path), exist_ok=True)
        
        if num_images == 1:
            outputs.images[0].save(output_image_path)
            print(f"图像已保存到 {output_image_path}")
        else:
            base_path = os.path.splitext(output_image_path)[0]
            extension = os.path.splitext(output_image_path)[1]
            for i, image in enumerate(outputs.images):
                path = f"{base_path}_{i}{extension}"
                image.save(path)
            print(f"已保存 {num_images} 张图像到 {os.path.dirname(output_image_path)}")
        
        return outputs.images
    except Exception as e:
        print(f"推理过程中发生错误: {e}")
        return None

if __name__ == "__main__":
    # 添加安装帮助命令
    parser = argparse.ArgumentParser(description="DreamBooth训练和推理")
    parser.add_argument("--install_help", action="store_true", help="显示安装指南")
    parser.add_argument("--train", action="store_true", help="训练模式")
    parser.add_argument("--infer", action="store_true", help="推理模式")
    parser.add_argument("--model_name", type=str, default=None, help="预训练模型名称")
    parser.add_argument("--small_model", action="store_true", help="使用小型模型")
    parser.add_argument("--model_path", type=str, default="./output", help="模型路径")
    parser.add_argument("--instance_data_dir", type=str, default="./instance_images", help="实例图像目录")
    parser.add_argument("--class_prompt", type=str, default="a dog", help="类别提示词")
    parser.add_argument("--identifier", type=str, help="手动指定稀有标识符")
    parser.add_argument("--prompt", type=str, help="推理时的完整提示词")
    parser.add_argument("--prior_weight", type=float, default=1.0, help="先验保留损失权重")
    parser.add_argument("--learning_rate", type=float, default=5e-6, help="学习率")
    parser.add_argument("--steps", type=int, default=800, help="训练步数")
    parser.add_argument("--train_text_encoder", action="store_true", help="是否训练文本编码器")
    parser.add_argument("--num_images", type=int, default=1, help="生成图像的数量")
    parser.add_argument("--prior_images", type=int, default=10, help="先验保留的类别图像数量")
    
    args = parser.parse_args()
    
    # 显示安装帮助
    if args.install_help:
        print("\n==== DreamBooth 安装指南 ====")
        print("\n推荐的安装方法:")
        print("1. 创建一个新的Conda环境:")
        print("   conda create -n dreambooth python=3.9")
        print("   conda activate dreambooth")
        print("\n2. 安装PyTorch (根据您的CUDA版本):")
        print("   # CUDA 11.8 (新的NVIDIA GPU):")
        print("   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
        print("   # CUDA 11.7:")
        print("   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117")
        print("   # CPU 或 macOS:")
        print("   pip install torch torchvision torchaudio")
        
        print("\n3. 安装兼容的依赖项:")
        print("   pip install diffusers==0.19.3 transformers==4.30.2 accelerate==0.21.0")
        print("   pip install Pillow tqdm numpy")
        
        print("\n4. 如果您有NVIDIA GPU，安装xformers以加速:")
        print("   pip install xformers")
        
        print("\n5. 如果仍然遇到问题，尝试这些兼容性更好的版本:")
        print("   pip install diffusers==0.14.0 transformers==4.25.1 accelerate==0.15.0")
        
        print("\n现在您可以运行DreamBooth:")
        print("   python dreambooth_implementation.py --train --small_model")
        exit(0)
    
    # 如果用户选择使用小模型或未指定模型名称，自动选择小模型
    if args.small_model or args.model_name is None:
        args.model_name = download_small_model()
        
    print(f"\n将使用模型: {args.model_name}")
    
    # 简化的小型模型训练提示
    if args.train:
        print("\n训练提示:")
        print("- 小模型约需 3-6GB 显存")
        print("- 如果内存不足，减少先验图像数量 (--prior_images 50)")
        print("- 如果仍然内存不足，可禁用文本编码器训练 (移除--train_text_encoder选项)")
        
        # 检查GPU并优化设置
        if torch.cuda.is_available():
            gpu_info = torch.cuda.get_device_name()
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"\n已检测到GPU: {gpu_info} ({gpu_memory:.1f}GB)")
            
            # 根据GPU内存调整优化参数
            if gpu_memory > 10:  # 高端GPU
                print("检测到高性能GPU，启用完整优化...")
            elif gpu_memory > 6:  # 中端GPU
                print("检测到中等性能GPU，启用部分优化...")
                if args.prior_images > 75:
                    print("提示: 考虑减少先验图像数量以加快速度 (--prior_images 75)")
            else:  # 低端GPU
                print("检测到低端GPU，使用最小资源配置...")
                if args.prior_images > 50:
                    print("注意: 由于GPU内存有限，建议减少先验图像数量 (--prior_images 30)")
                    if input("是否自动减少先验图像数量? [y/n]: ").lower() == 'y':
                        args.prior_images = 30
                        print("已调整先验图像数量为30")
        else:
            print("\n警告: 未检测到GPU! 训练将在CPU上运行，这会非常慢!")
            use_cpu = input("是否继续在CPU上训练? [y/n]: ").lower()
            if use_cpu != 'y':
                print("已取消训练。请在GPU环境下运行。")
                exit(0)
            
        _, identifier = dreambooth_training( # 接收返回值，虽然可能不用
            pretrained_model_name_or_path=args.model_name,
            instance_data_dir=args.instance_data_dir,
            output_dir=args.model_path,
            class_prompt=args.class_prompt,
            learning_rate=args.learning_rate,
            max_train_steps=args.steps,
            prior_preservation_weight=args.prior_weight,
            prior_generation_samples=args.prior_images,
            train_text_encoder=args.train_text_encoder,
        )
        # 添加完成消息
        if torch.cuda.is_available():
            print("\n训练过程完成。")
            if identifier:
                print(f"使用的标识符: {identifier}")
            print(f"模型保存在: {args.model_path}")
            
            # 询问用户是否要测试效果
            test_model = input("\n是否要立即测试模型效果? [y/n]: ").lower().strip()
            if test_model == 'y':
                print("\n正在测试模型效果...")
                class_name = args.class_prompt.replace("a ", "").strip()
                test_prompt = input(f"\n请输入提示词 (默认: a {identifier} {class_name} in a garden): ").strip()
                if not test_prompt:
                    test_prompt = f"a {identifier} {class_name} in a garden"
                
                # 调用inference进行测试
                inference(
                    model_path=args.model_path,
                    prompt=test_prompt,
                    class_prompt=args.class_prompt,
                    identifier=identifier,
                    num_images=1,
                    output_image_path="./test_result.png"
                )
                print("\n测试完成。如果要生成更多图像，请使用 --infer 参数运行程序。")

    elif args.infer:
        # 添加CUDA检测
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cpu":
            print("\n警告: 未检测到可用的CUDA GPU。将使用CPU运行，这会非常慢!")
            print("如果您的电脑有NVIDIA GPU，请确保:")
            print("1. 已安装正确的NVIDIA驱动程序")
            print("2. 已正确安装带CUDA支持的PyTorch")
            print("3. 如果您确定GPU可用，请尝试重启电脑")
            use_gpu = input("\n是否继续使用CPU? 这可能会非常慢 [y/n]: ").lower().strip()
            if use_gpu != 'y':
                print("程序已取消。请解决GPU问题后再尝试。")
                exit(0)
        else:
            gpu_info = torch.cuda.get_device_name()
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"\n成功检测到GPU: {gpu_info} ({gpu_memory:.1f}GB)")
            print(f"将使用设备: {device}")

        inference(
            model_path=args.model_path,
            prompt=args.prompt,
            class_prompt=args.class_prompt,
            identifier=args.identifier,
            num_images=args.num_images,
        )
    
    else:
        show_quick_help()

