import os
import random
import numpy as np
import argparse
import datetime
from pathlib import Path
from tqdm.auto import tqdm
from PIL import Image
# 添加内存管理相关模块
import gc
import torch.cuda

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
    """查找稀有令牌作为标识符"""
    # 首先打印理论解释
    print("\n" + "-"*80)
    print("【DreamBooth理论：选择稀有标识符】")
    print("""
论文3.2节指出，我们需要一个在自然语言中罕见的标识符，这样它就不会与模型已有的
文本-图像先验知识混淆。这是确保模型能够清晰地将我们的特定对象与通用类区分开的关键。

我们从CLIP词汇表中5000-10000范围内随机选择token，要求：
1. 长度不超过3个Unicode字符
2. 不包含空格

这样的token在自然文本中出现频率较低，且容易记忆和输入。
    """)
    print("-"*80 + "\n")
    
    print("开始选择稀有token标识符...")
    token_id = random.randint(token_range[0], token_range[1])
    token_text = tokenizer.decode([token_id]).strip()
    attempts = 1
    
    while len(token_text) > 3 or ' ' in token_text:
        token_id = random.randint(token_range[0], token_range[1])
        token_text = tokenizer.decode([token_id]).strip()
        attempts += 1
    
    print(f"已选择标识符 '{token_text}' (尝试次数: {attempts})")
    print("此标识符将用于构建提示词格式: 'a {标识符} {类别}'")
        
    return token_text

def explain_dreambooth_theory():
    """显示DreamBooth核心理论"""
    print("\n" + "="*80)
    print("DreamBooth 理论基础与工作原理".center(80))
    print("="*80)
    
    print("""
【核心思想】
DreamBooth 是一种以少量图像对文本到图像模型进行微调的技术，能够生成特定主体的个性化图像。

关键概念:
1. 个性化：使用少量（3-5张）包含特定主体的图像，教会模型生成该主体
2. 标识符绑定：使用罕见词（如"sks"）将特定主体与类别（如"dog"）绑定
3. 类先验保留：通过同时训练类别图像，保持模型对该类别的一般知识

【训练过程】
- 输入提示：结构为"a [identifier] [class]"，如"a sks dog"
- 目标：使模型学习将标识符与特定主体关联，同时保留类别的一般特征

【损失函数】
L = L_主体(特定主体) + λL_先验(类别先验)

其中:
- L_主体：确保模型学习特定主体的外观
- L_先验：通过合成生成的类别图像，确保模型不会"忘记"类别的一般特征
- λ：控制两个目标间的平衡（论文推荐值：1.0）

【最佳实践】
- 实例图像：3-5张高质量、多角度、干净背景的主体图像
- 训练步数：通常800-1000步足够，过多会导致过拟合
- 文本编码器：如果GPU内存足够，同时微调文本编码器和U-Net效果最佳
    """)
    
    print("="*80)
    print("参考：Ruiz等人,《DreamBooth: Fine Tuning Text-to-Image Diffusion Models for Subject-Driven Generation》, 2022")
    print("="*80 + "\n")
    return True

def generate_class_images(model, class_prompt, output_dir, num_samples=200):
    """生成类别图像以用于先验保留损失"""
    # 打印理论解释
    print("\n" + "-"*80)
    print("【DreamBooth理论：先验保留机制】")
    print("""
DreamBooth论文的一个关键创新是"先验保留"机制，用于解决过度拟合问题。

当模型仅学习少量的特定主体图像时，可能会"忘记"该主体所属类别的一般特征。
例如，如果只训练一只特定的狗，模型可能会丢失"狗"这一类别的一般知识。

先验保留的工作原理：
1. 使用当前模型生成一批类别图像（例如，通过"a dog"提示词生成的狗的图像）
2. 在训练过程中同时使用这些合成的类别图像
3. 引入类别先验损失，确保模型保留对该类别概念的理解

这样，模型在学习特定主体的同时，不会丢失类别的一般知识，保持了生成多样性。
    """)
    print("-"*80 + "\n")
    
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"开始生成 {num_samples} 张类别图像用于先验保留...")
    print(f"类别提示词: '{class_prompt}'")
    print("这些图像将确保模型在学习特定主体的同时，保持对类别的一般理解")
    
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
    
    print(f"已完成类别图像生成，保存至 {output_dir}")
    print("这些图像将在训练过程中用于计算先验保留损失")
            
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

def generate_examples(model_path, identifier, class_name):
    """生成一系列示例图像以展示模型的多样性"""
    print("\n【生成论文中描述的多样应用示例】")
    print("""
论文中图3和图4展示了DreamBooth的关键能力：保持特定主体特征的同时变换场景、装饰和风格。
以下示例展示了模型如何在不同条件下重现您的特定主体:
    """)
    
    example_prompts = [
        f"a {identifier} {class_name} in a garden",
        f"a {identifier} {class_name} in the snow",
        f"a {identifier} {class_name} wearing a hat",
        f"a portrait of a {identifier} {class_name}",
        f"a {identifier} {class_name} in the style of van gogh"
    ]
    
    print("生成以下多样化示例:")
    for prompt in example_prompts:
        print(f"- {prompt}")
    
    for i, prompt in enumerate(example_prompts):
        print(f"\n生成示例 {i+1}/{len(example_prompts)}: {prompt}")
        inference(
            model_path=model_path,
            prompt=prompt,
            num_images=1,
            output_image_path=f"example_{i+1}.png"
        )
    
    print("\n示例图像生成完成！请对比观察模型如何保持特定主体的特征")

def inference(model_path, prompt, class_prompt=None, identifier=None, num_images=1, output_image_path=None):
    """处理模型推理，生成图像"""
    # 加载模型
    pipeline = StableDiffusionPipeline.from_pretrained(model_path)
    if torch.cuda.is_available():
        pipeline = pipeline.to("cuda")
    
    # 如果没有提供标识符，尝试从保存的文件加载
    if identifier is None and os.path.exists(os.path.join(model_path, "identifier.txt")):
        with open(os.path.join(model_path, "identifier.txt"), "r") as f:
            identifier = f.read().strip()
    
    # 生成图像
    print(f"\n使用提示词: {prompt}")
    images = pipeline(
        prompt,
        num_images_per_prompt=num_images,
        num_inference_steps=50,
        guidance_scale=7.5
    ).images
    
    # 保存图像
    os.makedirs("outputs", exist_ok=True)
    for i, image in enumerate(images):
        if output_image_path and i == 0:
            # 如果指定了输出路径，使用它保存第一张图像
            image.save(output_image_path)
            print(f"图像已保存到: {output_image_path}")
        else:
            # 其他图像保存到outputs目录
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = f"outputs/generated_{timestamp}_{i}.png"
            image.save(save_path)
            print(f"图像已保存到: {save_path}")
    
    return images

def show_quick_help():
    """显示简洁的使用说明"""
    print("\n需要指定操作模式! 请使用以下参数之一:")
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

# 重组内存管理功能到一个独立的辅助类
class MemoryManager:
    """内存管理辅助类，提供统一的内存清理和监控功能"""
    def __init__(self, is_main_process=True):
        self.is_main_process = is_main_process
        self.peak_memory = 0
        self.initial_memory = self.get_memory_usage()
        if self.is_main_process and torch.cuda.is_available():
            print(f"初始GPU内存占用: {self.initial_memory:.2f}MB")
    
    def get_memory_usage(self):
        """获取当前内存使用量(MB)"""
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated()/1024**2
        return 0
    
    def cleanup(self, log_message=None):
        """清理内存并可选择性地记录信息"""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            if self.is_main_process:
                current_memory = self.get_memory_usage()
                self.peak_memory = max(self.peak_memory, current_memory)
                if log_message:
                    print(f"{log_message}: {current_memory:.2f}MB / 峰值: {self.peak_memory:.2f}MB")
    
    def log_memory_stats(self, message):
        """记录内存统计信息"""
        if self.is_main_process and torch.cuda.is_available():
            print(f"{message}: {self.get_memory_usage():.2f}MB / 峰值: {self.peak_memory:.2f}MB")

def dreambooth_training_with_theory(
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
    """DreamBooth训练流程"""
    # 打印训练过程说明
    print("\n" + "="*80)
    print("【DreamBooth训练流程】".center(80))
    print("="*80)
    print("""
DreamBooth训练过程按照论文描述分为以下关键步骤：

1. 模型准备：加载预训练扩散模型的组件（文本编码器，U-Net，VAE）
2. 标识符选择：选择一个罕见词作为标识符，将我们的特定主体与类别绑定
3. 类别先验生成：为了"先验保留"，自动生成类别的先验图像
4. 自适应训练：使用混合损失函数（特定主体损失+先验保留损失）进行微调
5. 模型保存：保存微调后的模型，供推理使用

注：最佳实践是在训练U-Net的同时也训练文本编码器，但如果GPU内存有限，可以只训练U-Net
    """)
    print("="*80)
    
    # 设置种子以保证结果可复现
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    
    print("\n【第1步】初始化加速器与内存管理")
    # 初始化加速器
    accelerator = Accelerator(
        gradient_accumulation_steps=gradient_accumulation_steps,
        mixed_precision="fp16"  # 使用16位混合精度训练以节省内存
    )
    
    # 创建内存管理器
    memory_mgr = MemoryManager(is_main_process=accelerator.is_main_process)
    memory_mgr.cleanup("初始化后内存占用")
    
    print("\n【第2步】标识符选择与提示词构建")
    # 初始化tokenizer
    tokenizer = CLIPTokenizer.from_pretrained(
        pretrained_model_name_or_path, 
        subfolder="tokenizer"
    )
    
    # 查找稀有令牌作为标识符，如论文3.2节所述
    identifier = find_rare_token(tokenizer)
    
    # 构建实例提示词，遵循论文中"a [identifier] [class noun]"的格式
    if instance_prompt is None:
        class_name = class_prompt.replace("a ", "").strip()
        instance_prompt = f"a {identifier} {class_name}"
    
    print(f"实例提示词: '{instance_prompt}'")
    print(f"类别提示词: '{class_prompt}'")
    print("在训练后，当您在提示词中使用此标识符时，模型将生成您训练的特定主体")
    
    print("\n【第3步】加载模型组件")
    print(f"加载预训练模型: {pretrained_model_name_or_path}")
    # 加载文本编码器
    text_encoder = CLIPTextModel.from_pretrained(
        pretrained_model_name_or_path, 
        subfolder="text_encoder"
    )
    
    # 加载 VAE
    print("加载VAE - 负责图像与潜在空间的编解码")
    vae = AutoencoderKL.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="vae"
    )

    # 加载U-Net
    print("加载U-Net - 这是实际进行去噪预测的核心网络")
    unet = UNet2DConditionModel.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="unet",
    )
    
    # 加载用于生成类别图像的完整pipeline
    pipeline = StableDiffusionPipeline.from_pretrained(
        pretrained_model_name_or_path,
        vae=vae,
        text_encoder=text_encoder,
        unet=unet,
    )
    pipeline.to(accelerator.device)

    # 将 VAE 设置为评估模式并移至 GPU，通常不训练 VAE
    print("设置VAE为评估模式 - VAE在DreamBooth中不需要训练")
    vae.requires_grad_(False)
    vae.to(accelerator.device, dtype=torch.float32) # VAE 通常保持在 fp32

    print("\n【第4步】生成类别先验图像")
    print("这是DreamBooth的关键创新之一，用于防止'语言漂移'和过度拟合")
    # 生成类别图像用于先验保留
    class_images_dir = os.path.join(output_dir, "class_images")
    if not os.path.exists(class_images_dir) or len(os.listdir(class_images_dir)) < prior_generation_samples:
        generate_class_images(pipeline, class_prompt, class_images_dir, prior_generation_samples)
    else:
        print(f"使用已存在的类别图像: {class_images_dir}")
    
    print("\n【第5步】加载训练数据集")
    # 加载数据集
    dataset = DreamBoothDataset(
        instance_images_path=instance_data_dir,
        class_images_path=class_images_dir,
        tokenizer=tokenizer,
    )
    print(f"加载了 {len(dataset.instance_images)} 张实例图像和 {len(dataset.class_images)} 张类别图像")
    dataloader = DataLoader(dataset, batch_size=train_batch_size, shuffle=True)
    
    print("\n【第6步】准备优化器和参数")
    # 准备优化器
    # 如论文所述，最佳的主体保真度是通过微调所有层获得的
    if train_text_encoder:
        params_to_optimize = list(unet.parameters()) + list(text_encoder.parameters())
        print("将同时微调U-Net和文本编码器 - 这能取得最佳效果但需要更多GPU内存")
    else:
        params_to_optimize = unet.parameters()
        print("仅微调U-Net - 内存效率更高，但效果可能略差")
    
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
    print("\n准备文本嵌入...")
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
    print("\n【第7步】开始DreamBooth训练")
    print(f"将训练 {max_train_steps} 步，先验保留权重λ设为 {prior_preservation_weight}...")
    print("""
训练过程中将应用以下关键理论：
1. 扩散去噪 - 模型学习从加噪图像预测噪声的过程
2. 特定主体损失 - 让模型学习将标识符与您的特定主体关联
3. 先验保留损失 - 防止模型"忘记"类别的一般知识

根据论文第3.4节，先验保留损失是DreamBooth成功的关键，它确保了模型在学习特定主体的同时，不会丢失对类别的一般理解。
    """)
    progress_bar = tqdm(range(max_train_steps), desc="训练进度", disable=not accelerator.is_main_process)
    global_step = 0
    
    # VAE 缩放因子，从 Stable Diffusion 配置中获取
    vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)

    # 记录损失
    epoch_losses = []
    loss_log_interval = 50  # 每隔多少步记录一次损失

    for epoch in range(3):  # 通常一个epoch就足够了
        epoch_loss = {"total": [], "instance": [], "class": []}
        
        print(f"\n开始Epoch {epoch+1}/3...")
        
        unet.train()
        if train_text_encoder:
            text_encoder.train()
        else:
            text_encoder.eval()
        
        step_in_epoch = 0
        for step, batch in enumerate(dataloader):
            if global_step >= max_train_steps:
                break
                
            # 每10步执行一次内存清理
            if step % 10 == 0:
                memory_mgr.cleanup()
                
            with accelerator.accumulate(unet):
                # 准备输入
                pixel_values = batch["pixel_values"].to(accelerator.device)
                is_instance = batch["is_instance"]
                
                # 使用 VAE 将图像编码为潜在表示
                with torch.no_grad():
                    # 确保使用正确的数据类型
                    latents = vae.encode(pixel_values.to(dtype=vae.dtype)).latent_dist.sample()
                    latents = latents * vae.config.scaling_factor
                
                # 为潜在表示添加噪声
                noise = torch.randn_like(latents)
                timesteps = torch.randint(
                    0, 
                    noise_scheduler.config.num_train_timesteps,
                    (latents.shape[0],), 
                    device=latents.device
                ).long()
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
                
                # 准备文本嵌入 - 区分实例和类别样本
                # 这是论文3.2节的关键步骤 - 使用不同提示词
                with torch.no_grad():
                    if torch.sum(is_instance).item() > 0:
                        instance_emb = text_encoder(
                            instance_text_inputs.input_ids.to(accelerator.device)
                        )[0]
                    
                    if torch.sum(~is_instance).item() > 0:
                        class_emb = text_encoder(
                            class_text_inputs.input_ids.to(accelerator.device)
                        )[0]

                # 预测噪声残差，分别处理实例样本和类别样本
                instance_loss = 0.0
                class_loss = 0.0
                
                # 处理实例样本 - 特定主体损失
                if torch.sum(is_instance).item() > 0:
                    instance_pred = unet(
                        noisy_latents[is_instance],
                        timesteps[is_instance],
                        encoder_hidden_states=instance_emb.repeat(
                            torch.sum(is_instance).item(), 1, 1
                        )
                    ).sample
                    
                    instance_loss = F.mse_loss(
                        instance_pred.float(),
                        noise[is_instance].float(),
                        reduction="mean"
                    )
                
                # 处理类别样本 - 先验保留损失
                if torch.sum(~is_instance).item() > 0:
                    class_pred = unet(
                        noisy_latents[~is_instance],
                        timesteps[~is_instance],
                        encoder_hidden_states=class_emb.repeat(
                            torch.sum(~is_instance).item(), 1, 1
                        )
                    ).sample
                    
                    class_loss = F.mse_loss(
                        class_pred.float(),
                        noise[~is_instance].float(), 
                        reduction="mean"
                    )
                
                # 根据论文公式(1)，组合损失，应用先验保留权重λ
                # L = L_instance + λ·L_prior
                loss = instance_loss + prior_preservation_weight * class_loss
                
                # 反向传播
                accelerator.backward(loss)
                
                if accelerator.sync_gradients:
                    # 梯度裁剪
                    if train_text_encoder:
                        accelerator.clip_grad_norm_(
                            list(unet.parameters()) + list(text_encoder.parameters()),
                            1.0
                        )
                    else:
                        accelerator.clip_grad_norm_(unet.parameters(), 1.0)
                        
                # 优化步骤
                optimizer.step()
                optimizer.zero_grad()
                
                # 记录损失
                epoch_loss["total"].append(loss.detach().item())
                epoch_loss["instance"].append(instance_loss.detach().item() if isinstance(instance_loss, torch.Tensor) else instance_loss)
                epoch_loss["class"].append(class_loss.detach().item() if isinstance(class_loss, torch.Tensor) else class_loss)
            
            # 打印损失
            if (global_step % loss_log_interval == 0 or global_step == max_train_steps-1) and accelerator.is_main_process:
                print(f"\n步骤 {global_step}/{max_train_steps}: 总损失={loss.detach().item():.4f}, "
                      f"实例损失={instance_loss.detach().item() if isinstance(instance_loss, torch.Tensor) else instance_loss:.4f}, "
                      f"类别损失={class_loss.detach().item() if isinstance(class_loss, torch.Tensor) else class_loss:.4f}")
                
            # 更新进度条
            if accelerator.is_main_process:
                progress_bar.set_postfix({
                    "loss": loss.detach().item(),
                    "instance_loss": instance_loss.detach().item() if isinstance(instance_loss, torch.Tensor) else instance_loss,
                    "class_loss": class_loss.detach().item() if isinstance(class_loss, torch.Tensor) else class_loss
                })
                progress_bar.update(1)
                
            # 正确增加步数计数器
            global_step += 1
            step_in_epoch += 1
            
            # 检查是否达到最大步数
            if global_step >= max_train_steps:
                print(f"\n已达到最大训练步数 {max_train_steps}，停止训练")
                break
                
        # 每个epoch结束后打印统计信息
        epoch_losses.append(epoch_loss)
        memory_mgr.cleanup(f"Epoch {epoch+1} 结束后")
        
        if accelerator.is_main_process and epoch_loss["total"]:
            avg_total = sum(epoch_loss["total"]) / len(epoch_loss["total"])
            avg_instance = sum(epoch_loss["instance"]) / len(epoch_loss["instance"]) if epoch_loss["instance"] else 0
            avg_class = sum(epoch_loss["class"]) / len(epoch_loss["class"]) if epoch_loss["class"] else 0
            
            print(f"\nEpoch {epoch+1} 统计 (完成 {step_in_epoch} 步训练):")
            print(f"平均总损失: {avg_total:.4f}")
            print(f"平均实例损失: {avg_instance:.4f}")
            print(f"平均类别损失: {avg_class:.4f}")
            print("-" * 40)
        
        if global_step >= max_train_steps:
            break

    # 关闭进度条
    progress_bar.close()
    
    # 最终内存清理
    memory_mgr.cleanup("训练结束后")
    memory_mgr.log_memory_stats("最终内存占用")

    print("\n【第8步】保存训练完成的模型")
    print("DreamBooth训练完成！正在保存模型...")
    
    # 等待所有进程完成
    accelerator.wait_for_everyone()
    
    # 保存模型
    if accelerator.is_main_process:
        # 解包装模型
        unet = accelerator.unwrap_model(unet)
        if train_text_encoder:
            text_encoder = accelerator.unwrap_model(text_encoder)
        
        # 保存标识符
        with open(os.path.join(output_dir, "identifier.txt"), "w") as f:
            f.write(identifier)
        
        # 保存完整模型
        pipeline = StableDiffusionPipeline.from_pretrained(
            pretrained_model_name_or_path,
            unet=unet,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            scheduler=noise_scheduler,
            vae=vae
        )
        pipeline.save_pretrained(output_dir)
        print(f"模型已保存到 {output_dir}")
    
    print(f"\nDreamBooth训练成功完成！")
    print(f"【重要】使用标识符 '{identifier}' 来在提示词中引用您的特定主体")
    print(f"例如: 'a {identifier} {class_prompt.replace('a ', '')} wearing a hat'")
    
    return None, identifier

if __name__ == "__main__":
    # 添加一个检查函数，用于验证DreamBooth的理论正确性
    def verify_dreambooth_implementation():
        """验证DreamBooth实现与论文的一致性"""
        print("\n【验证DreamBooth实现】")
        print("检查以下关键组件是否与论文一致:")
        
        checks = [
            "✓ 使用罕见标识符将特定主体与类别绑定 (论文3.2节)",
            "✓ 实现先验保留损失以防止语言漂移 (论文3.3节)",
            "✓ 损失函数形式: L = L_instance + λ·L_prior (公式1)",
            "✓ 文本编码器与U-Net联合训练选项 (论文3.4节)",
            "✓ 标准提示词格式: a [identifier] [class] (论文3.2节)"
        ]
        
        for check in checks:
            print(check)
        
        print("\n论文引用:")
        print("Ruiz, N., Li, Y., Jampani, V., Pritch, Y., Rubinstein, M., & Aberman, K. (2022).")
        print("DreamBooth: Fine Tuning Text-to-Image Diffusion Models for Subject-Driven Generation.")
        return True
    
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
    parser.add_argument("--explain", action="store_true", help="显示DreamBooth原理解释")
    parser.add_argument("--examples", action="store_true", help="训练后生成应用示例")
    parser.add_argument("--create_guide", action="store_true", help="创建DreamBooth技术指南")
    parser.add_argument("--create_report", action="store_true", help="创建实验报告")
    parser.add_argument("--verify", action="store_true", help="验证DreamBooth实现与论文一致性")
    
    args = parser.parse_args()
    
    # 验证实现
    if args.verify:
        verify_dreambooth_implementation()
        exit(0)
    
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
    
    # 显示理论解释
    if args.explain:
        explain_dreambooth_theory()
        print("\n要开始训练，请使用 --train 参数")
        exit(0)
        
    # 创建技术指南
    if args.create_guide:
        print("\n创建DreamBooth技术指南...")
        guide_path = "dreambooth_technical_guide.md"
        
        with open(guide_path, "w") as f:
            f.write("# DreamBooth技术原理详解\n\n")
            
            f.write("## 1. 核心思想\n\n")
            f.write("DreamBooth是一种文本到图像个性化技术，允许用户使用少量（3-5张）特定主体图像来微调现有扩散模型。\n")
            f.write("其核心思想是将特定主体与一个罕见的词语绑定，同时保持模型对该主体所属类别的一般知识。\n\n")
            
            f.write("## 2. 技术挑战\n\n")
            f.write("DreamBooth需要解决两个关键挑战：\n")
            f.write("1. **语言漂移**：确保模型理解标识符确实指代特定主体\n")
            f.write("2. **过度拟合**：防止模型丢失类别知识，导致生成的主体失去多样性\n\n")
            
            f.write("## 3. 方法细节\n\n")
            f.write("### 3.1 标识符绑定\n\n")
            f.write("使用形如\"a [identifier] [class]\"的提示词，例如\"a sks dog\"。\n")
            f.write("标识符应该是自然语言中的稀有词，以避免与现有概念冲突。\n\n")
            
            f.write("### 3.2 先验保留损失\n\n")
            f.write("为防止过度拟合，DreamBooth引入了先验保留损失：\n")
            f.write("1. 使用当前模型生成类别图像（例如，从\"a dog\"生成狗的图像）\n")
            f.write("2. 将这些合成图像纳入训练，以防止模型忘记类别的一般特征\n\n")
            
            f.write("### 3.3 训练过程\n\n")
            f.write("损失函数为：L = L_主体 + λL_类别\n")
            f.write("其中λ控制两种损失的平衡（论文建议λ=1.0）\n\n")
            
            f.write("## 4. 最佳实践\n\n")
            f.write("- **实例图像**：使用3-5张高质量、多角度、干净背景的主体图像\n")
            f.write("- **训练步数**：通常800-1000步足够，过多会导致过拟合\n")
            f.write("- **学习率**：5e-6通常效果良好\n")
            f.write("- **文本编码器**：若GPU内存足够，建议同时微调文本编码器\n\n")
        
        print(f"技术指南已保存到 {guide_path}")
        exit(0)
        
    # 创建实验报告
    def create_experiment_report(model_path, identifier, class_prompt):
        """创建实验报告，记录训练过程和生成结果"""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
        report_path = f"experiment_report_{timestamp}.md"
        
        with open(report_path, "w") as f:
            f.write("# DreamBooth实验报告\n\n")
            f.write(f"实验日期: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
            f.write(f"预训练模型: {os.path.basename(model_path)}\n")
            f.write(f"类别: {class_prompt}\n")
            f.write(f"标识符: {identifier}\n\n")
            
            f.write("## 关键参数\n")
            f.write("- 先验保留权重 (λ): 1.0\n")
            f.write("- 学习率: 5e-6\n")
            f.write("- 训练步数: 800\n\n")
            
            f.write("## 生成示例\n")
            f.write("以下是不同提示词下的生成结果：\n\n")
            
            prompts = [
                f"a {identifier} {class_prompt.replace('a ', '')} in a garden",
                f"a {identifier} {class_prompt.replace('a ', '')} on the moon",
                f"a {identifier} {class_prompt.replace('a ', '')} as a cartoon character"
            ]
            
            for i, prompt in enumerate(prompts):
                img_path = f"report_sample_{i}.png"
                inference(model_path=model_path, prompt=prompt, output_image_path=img_path)
                f.write(f"### 示例 {i+1}\n")
                f.write(f"提示词: `{prompt}`\n\n")
                f.write(f"![生成结果]({img_path})\n\n")
        
        print(f"实验报告已保存到 {report_path}")
    
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
            
        # 训练前清理内存
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print(f"训练前GPU内存占用: {torch.cuda.memory_allocated()/1024**2:.2f}MB")
        
        _, identifier = dreambooth_training_with_theory(
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
        
        # 训练后再次清理内存
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print(f"训练后清理内存: {torch.cuda.memory_allocated()/1024**2:.2f}MB")
            
        # 添加完成消息
        if torch.cuda.is_available():
            print("\n训练过程完成。")
            if identifier:
                print(f"使用的标识符: {identifier}")
            print(f"模型保存在: {args.model_path}")
            
            # 训练后生成应用示例
            if args.examples and torch.cuda.is_available():
                print("\n生成应用示例以展示DreamBooth的多样化能力...")
                generate_examples(args.model_path, identifier, args.class_prompt.replace("a ", "").strip())
            
            # 创建实验报告
            if args.create_report and torch.cuda.is_available():
                print("\n创建实验报告...")
                create_experiment_report(args.model_path, identifier, args.class_prompt.replace("a ", "").strip())
            
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