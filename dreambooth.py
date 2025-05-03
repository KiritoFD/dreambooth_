import os
import random
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm
from PIL import Image

from diffusers import StableDiffusionPipeline, DDPMScheduler, UNet2DConditionModel, AutoencoderKL
from transformers import CLIPTextModel, CLIPTokenizer
from accelerate import Accelerator

# 导入理论笔记用于训练过程中打印
try:
    from theory_notes import DreamBoothTheory, get_theory_step, get_training_step, print_theory_step
    HAS_THEORY_NOTES = True
    print("已加载DreamBooth理论笔记模块，将在训练过程中提供论文解析...")
except ImportError:
    HAS_THEORY_NOTES = False
    print("未找到理论笔记模块，将只显示基本训练信息...")

class DreamBoothDataset(Dataset):
    def __init__(self, instance_images_path, class_images_path, tokenizer, size=512, center_crop=True):
        self.size = size
        self.center_crop = center_crop
        self.tokenizer = tokenizer
        
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
                    if self.center_crop:
                        w, h = img.size
                        min_dim = min(w, h)
                        img = img.crop(((w-min_dim)//2, (h-min_dim)//2, (w+min_dim)//2, (h+min_dim)//2))
                    img = img.resize((self.size, self.size))
                    images.append(img)
                except Exception:
                    pass
        return images
    
    def __len__(self):
        return len(self.instance_images) + len(self.class_images)
    
    def __getitem__(self, idx):
        if idx < len(self.instance_images):
            image = self.instance_images[idx]
            is_instance = True
        else:
            image = self.class_images[idx - len(self.instance_images)]
            is_instance = False
        
        image = torch.from_numpy(np.array(image).astype(np.float32) / 127.5 - 1.0)
        image = image.permute(2, 0, 1)
        
        return {"pixel_values": image, "is_instance": is_instance}

def find_rare_token(tokenizer, token_range=(5000, 10000)):
    # 打印标识符选择理论
    if HAS_THEORY_NOTES:
        theory = get_theory_step("initialization")
        if theory:
            print_theory_step("1", theory["title"], theory["description"])
    
    token_id = random.randint(token_range[0], token_range[1])
    token_text = tokenizer.decode([token_id]).strip()
    attempts = 1
    
    while len(token_text) > 3 or ' ' in token_text:
        token_id = random.randint(token_range[0], token_range[1])
        token_text = tokenizer.decode([token_id]).strip()
        attempts += 1
    
    print(f"已选择标识符 '{token_text}' (尝试次数: {attempts})")
    return token_text

def generate_class_images(model, class_prompt, output_dir, num_samples=200):
    # 打印先验保留理论
    if HAS_THEORY_NOTES:
        theory = get_theory_step("prior_preservation")
        if theory:
            print_theory_step("2", theory["title"], theory["description"])
    
    os.makedirs(output_dir, exist_ok=True)
    model.safety_checker = None
    
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

def dreambooth_training(
    pretrained_model_name_or_path,
    instance_data_dir,
    output_dir,
    class_prompt,
    instance_prompt=None,
    learning_rate=5e-6,
    max_train_steps=1000,
    prior_preservation_weight=1.0,
    prior_generation_samples=200,
    gradient_accumulation_steps=1,
    train_text_encoder=True,
    train_batch_size=1,
    seed=42,
    memory_mgr=None,
):
    """DreamBooth核心训练逻辑"""
    # 分阶段执行训练流程，每个阶段都有对应的理论解释
    training_stages = [
        "准备工作",             # 第1阶段：环境准备和初始设置
        "模型加载",             # 第2阶段：加载预训练模型
        "标识符选择",           # 第3阶段：选择和绑定稀有标识符 
        "先验图像生成",         # 第4阶段：生成类别先验图像
        "数据集构建",           # 第5阶段：构建训练数据集
        "优化器配置",           # 第6阶段：设置优化器和训练参数
        "训练循环开始",         # 第7阶段：开始主训练循环
        "训练循环中期",         # 第8阶段：训练进行中
        "训练循环结束",         # 第9阶段：完成训练
        "模型保存",             # 第10阶段：保存训练结果
        "应用建议"              # 第11阶段：推理应用指导
    ]
    
    current_stage = 0
    def update_stage():
        nonlocal current_stage
        current_stage += 1
        if HAS_THEORY_NOTES:
            print("\n" + "="*80)
            print(f"【{training_stages[current_stage-1]}】- 第{current_stage}/{len(training_stages)}阶段".center(80))
            print("="*80)
    
    # 第1阶段：准备工作
    update_stage()
    if HAS_THEORY_NOTES:
        print(DreamBoothTheory.core_problem())
    
    # 设置随机种子和环境
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    
    if HAS_THEORY_NOTES:
        step_info = get_training_step("initialization")
        if step_info:
            print("\n" + "-"*60)
            print("初始化与环境准备")
            print("-"*60)
            print("DreamBooth训练需要以下准备工作：\n")
            print("1. 设置随机种子以确保结果可复现")
            print("2. 配置加速器以优化训练过程")
            print("3. 准备输入输出目录结构")
    
    # 初始化加速器
    accelerator = Accelerator(
        gradient_accumulation_steps=gradient_accumulation_steps,
        mixed_precision="fp16"
    )
    
    # 创建必要的目录
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "class_images"), exist_ok=True)
    
    # 第2阶段：模型加载
    update_stage()
    if HAS_THEORY_NOTES:
        print("\n" + "-"*60)
        print("加载预训练模型组件")
        print("-"*60)
        print("DreamBooth基于预训练的扩散模型，需要加载以下组件：\n")
        print("1. 文本编码器 (Text Encoder) - 将文本转换为嵌入")
        print("2. 自编码器 (VAE) - 图像压缩与重建")
        print("3. U-Net - 核心扩散模型网络")
        print("4. 分词器 (Tokenizer) - 文本处理")
    
    # 加载分词器
    tokenizer = CLIPTokenizer.from_pretrained(
        pretrained_model_name_or_path, 
        subfolder="tokenizer"
    )
    
    # 加载模型组件
    text_encoder = CLIPTextModel.from_pretrained(
        pretrained_model_name_or_path, subfolder="text_encoder"
    )
    vae = AutoencoderKL.from_pretrained(
        pretrained_model_name_or_path, subfolder="vae"
    )
    unet = UNet2DConditionModel.from_pretrained(
        pretrained_model_name_or_path, subfolder="unet"
    )
    
    # 将VAE设置为评估模式（不训练）
    vae.requires_grad_(False)
    vae.to(accelerator.device, dtype=torch.float32)
    
    # 第3阶段：标识符选择
    update_stage()
    # 选择稀有标识符和构建提示词
    identifier = find_rare_token(tokenizer)
    
    if instance_prompt is None:
        class_name = class_prompt.replace("a ", "").strip()
        instance_prompt = f"a {identifier} {class_name}"
    
    print(f"实例提示词: '{instance_prompt}'")
    print(f"类别提示词: '{class_prompt}'")
    
    if HAS_THEORY_NOTES:
        print("\n" + "-"*60)
        print("标识符绑定机制详解")
        print("-"*60)
        print(f"您的特定主体现在与标识符 '{identifier}' 绑定")
        print(f"实例提示词 '{instance_prompt}' 将用于训练特定主体表示")
        print(f"类别提示词 '{class_prompt}' 将用于生成先验保留图像")
    
    # 第4阶段：先验图像生成
    update_stage()
    # 生成类别先验图像 - 打印相关理论信息
    if HAS_THEORY_NOTES:
        step_info = get_training_step("prior_image_generation")
        if step_info:
            print("\n" + "-"*60)
            print("类别先验图像生成")
            print("-"*60)
            print(step_info["description"])
    
    # 加载生成管道
    pipeline = StableDiffusionPipeline.from_pretrained(
        pretrained_model_name_or_path,
        vae=vae, text_encoder=text_encoder, unet=unet,
    )
    pipeline.to(accelerator.device)
    
    # 生成类别先验图像
    class_images_dir = os.path.join(output_dir, "class_images")
    if not os.path.exists(class_images_dir) or len(os.listdir(class_images_dir)) < prior_generation_samples:
        print(f"\n开始生成{prior_generation_samples}张类别图像用于先验保留...")
        generate_class_images(pipeline, class_prompt, class_images_dir, prior_generation_samples)
    else:
        existing_images = len([f for f in os.listdir(class_images_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
        print(f"\n在'{class_images_dir}'目录中找到{existing_images}张现有类别图像，跳过生成步骤")
    
    # 释放生成管道以节省内存
    del pipeline
    if memory_mgr:
        memory_mgr.cleanup("释放生成管道后")
    
    # 第5阶段：数据集构建
    update_stage()
    if HAS_THEORY_NOTES:
        step_info = get_training_step("dataset_preparation")
        if step_info:
            print("\n" + "-"*60)
            print("训练数据集构建")
            print("-"*60)
            print(step_info["description"])
    
    # 加载并检查实例图像
    instance_images_count = len([f for f in os.listdir(instance_data_dir) 
                              if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    if instance_images_count == 0:
        raise ValueError(f"错误: 在'{instance_data_dir}'目录中未找到任何图像文件")
    elif instance_images_count < 3:
        print(f"警告: 仅发现{instance_images_count}张实例图像。DreamBooth建议使用3-5张图像以获得最佳效果")
    else:
        print(f"已发现{instance_images_count}张实例图像，符合DreamBooth推荐的3-5张图像范围")
    
    # 创建数据集
    dataset = DreamBoothDataset(
        instance_images_path=instance_data_dir,
        class_images_path=class_images_dir,
        tokenizer=tokenizer,
    )
    print(f"创建了训练数据集，包含{len(dataset.instance_images)}张实例图像和{len(dataset.class_images)}张类别图像")
    
    # 创建数据加载器
    dataloader = DataLoader(dataset, batch_size=train_batch_size, shuffle=True)
    
    # 第6阶段：优化器配置
    update_stage()
    if HAS_THEORY_NOTES:
        step_info = get_training_step("optimization_setup")
        if step_info:
            print("\n" + "-"*60)
            print("优化器与训练参数配置")
            print("-"*60)
            print(step_info["description"])
        
        # 打印训练策略理论
        theory = get_theory_step("training")
        if theory:
            print_theory_step("3", theory["title"], theory["description"])
    
    # 准备优化器
    params_to_optimize = []
    if train_text_encoder:
        print("将同时优化U-Net和文本编码器")
        params_to_optimize = list(unet.parameters()) + list(text_encoder.parameters())
    else:
        print("仅优化U-Net (文本编码器保持冻结)")
        params_to_optimize = list(unet.parameters())
    
    optimizer = torch.optim.AdamW(
        params_to_optimize,
        lr=learning_rate,
        betas=(0.9, 0.999),
        weight_decay=1e-2,
    )
    
    # 准备噪声调度器
    noise_scheduler = DDPMScheduler.from_pretrained(
        pretrained_model_name_or_path, subfolder="scheduler"
    )
    
    # 准备文本嵌入
    print("生成文本条件嵌入...")
    instance_text_inputs = tokenizer(
        instance_prompt, padding="max_length", 
        max_length=tokenizer.model_max_length,
        truncation=True, return_tensors="pt"
    )
    class_text_inputs = tokenizer(
        class_prompt, padding="max_length",
        max_length=tokenizer.model_max_length, 
        truncation=True, return_tensors="pt"
    )
    
    # 准备加速训练
    print("准备加速训练...")
    unet, text_encoder, optimizer, dataloader = accelerator.prepare(
        unet, text_encoder, optimizer, dataloader
    )
    
    # 第7阶段：训练循环开始
    update_stage()
    if HAS_THEORY_NOTES:
        step_info = get_training_step("training_loop")
        if step_info:
            print("\n" + "-"*60)
            print("训练循环开始")
            print("-"*60)
            print(step_info["description"])
        
        # 打印损失函数理论
        theory = get_theory_step("loss_function")
        if theory:
            print_theory_step("4", theory["title"], theory["description"])
    
    progress_bar = tqdm(range(max_train_steps), desc="训练进度")
    global_step = 0
    
    # 检查点恢复逻辑
    checkpoint_path = os.path.join(output_dir, "checkpoint.pt")
    start_step = 0
    
    if os.path.exists(checkpoint_path):
        print(f"发现检查点，尝试恢复训练...")
        try:
            checkpoint = torch.load(checkpoint_path)
            global_step = checkpoint["global_step"]
            start_step = global_step
            optimizer.load_state_dict(checkpoint["optimizer"])
            unet.load_state_dict(checkpoint["unet"])
            if train_text_encoder and "text_encoder" in checkpoint:
                text_encoder.load_state_dict(checkpoint["text_encoder"])
            print(f"成功恢复训练，从步骤 {global_step} 开始")
            
            # 更新进度条
            progress_bar = tqdm(range(global_step, max_train_steps), 
                              initial=global_step, total=max_train_steps,
                              desc="训练进度")
        except Exception as e:
            print(f"恢复训练失败: {str(e)}")
            print("将从头开始训练")
    
    # 训练过程监控变量
    early_stop_threshold = 3
    no_improvement_count = 0
    last_loss = float('inf')
    save_checkpoint_steps = 50
    
    # 记录损失值
    loss_history = {
        "instance": [],
        "class": [],
        "total": [],
        "steps": []
    }
    
    # 主要训练循环
    try:
        for epoch in range(1):  # 通常一个epoch就足够
            unet.train()
            text_encoder.train() if train_text_encoder else text_encoder.eval()
            
            # 从检查点步骤开始迭代
            for step, batch in enumerate(dataloader):
                if step < start_step % len(dataloader):
                    continue
                
                if global_step >= max_train_steps:
                    break
                    
                # 内存清理
                if memory_mgr and step % 10 == 0:
                    memory_mgr.cleanup()
                
                # 到达中期阶段，打印训练状态
                if global_step == max_train_steps // 2 and global_step > 0:
                    # 第8阶段：训练循环中期
                    update_stage()
                    if HAS_THEORY_NOTES:
                        print("\n" + "-"*60)
                        print(f"训练已完成50%: {global_step}/{max_train_steps}步")
                        print("-"*60)
                        print("此时模型已开始学习特定主体特征，但需要继续训练以获得最佳效果")
                        print("根据论文3.4节，通常需要800-1000步才能充分学习主体特征但又不过度拟合")
                
                with accelerator.accumulate(unet):
                    # 准备输入
                    pixel_values = batch["pixel_values"].to(accelerator.device)
                    is_instance = batch["is_instance"]
                    
                    # 编码到潜在空间
                    with torch.no_grad():
                        latents = vae.encode(pixel_values.to(dtype=vae.dtype)).latent_dist.sample()
                        latents = latents * vae.config.scaling_factor
                    
                    # 添加噪声
                    noise = torch.randn_like(latents)
                    timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, 
                                              (latents.shape[0],), device=latents.device).long()
                    noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
                    
                    # 准备文本嵌入
                    with torch.no_grad():
                        instance_emb = text_encoder(
                            instance_text_inputs.input_ids.to(accelerator.device)
                        )[0] if torch.sum(is_instance).item() > 0 else None
                        
                        class_emb = text_encoder(
                            class_text_inputs.input_ids.to(accelerator.device)
                        )[0] if torch.sum(~is_instance).item() > 0 else None

                    # 计算损失
                    instance_loss = 0.0
                    class_loss = 0.0
                    
                    # 实例损失(特定主体)
                    if torch.sum(is_instance).item() > 0:
                        instance_pred = unet(
                            noisy_latents[is_instance],
                            timesteps[is_instance],
                            encoder_hidden_states=instance_emb.repeat(torch.sum(is_instance).item(), 1, 1)
                        ).sample
                        
                        instance_loss = F.mse_loss(
                            instance_pred.float(),
                            noise[is_instance].float(),
                            reduction="mean"
                        )
                    
                    # 类别损失(先验保留)
                    if torch.sum(~is_instance).item() > 0:
                        class_pred = unet(
                            noisy_latents[~is_instance],
                            timesteps[~is_instance],
                            encoder_hidden_states=class_emb.repeat(torch.sum(~is_instance).item(), 1, 1)
                        ).sample
                        
                        class_loss = F.mse_loss(
                            class_pred.float(),
                            noise[~is_instance].float(),
                            reduction="mean"
                        )
                    
                    # 组合损失 (论文公式1)
                    loss = instance_loss + prior_preservation_weight * class_loss
                    
                    # 记录损失
                    if accelerator.is_main_process and global_step % 10 == 0:
                        il = instance_loss.detach().item() if isinstance(instance_loss, torch.Tensor) else instance_loss
                        cl = class_loss.detach().item() if isinstance(class_loss, torch.Tensor) else class_loss
                        tl = loss.detach().item()
                        loss_history["instance"].append(il)
                        loss_history["class"].append(cl)
                        loss_history["total"].append(tl)
                        loss_history["steps"].append(global_step)
                    
                    # 每200步打印训练详细状态
                    if global_step % 200 == 0 and global_step > 0 and HAS_THEORY_NOTES:
                        print("\n" + "-"*60)
                        print(f"【训练进度详情 - 步骤 {global_step}/{max_train_steps}】")
                        print("-"*60)
                        print(f"""
训练理论对应关系:
- 实例损失对应论文公式(1)中的L_instance项
- 类别损失对应论文公式(1)中的L_prior项
- 总损失为: L = L_instance + {prior_preservation_weight} · L_prior

训练进度: {global_step/max_train_steps*100:.1f}% 完成

当前步骤损失值:
- 实例损失: {instance_loss.detach().item() if isinstance(instance_loss, torch.Tensor) else instance_loss:.6f}
- 类别损失: {class_loss.detach().item() if isinstance(class_loss, torch.Tensor) else class_loss:.6f}
- 总损失: {loss.detach().item():.6f}
""")
                    
                    # 检测损失是否有改善
                    current_loss = loss.detach().item()
                    if abs(current_loss - last_loss) < 1e-5:
                        no_improvement_count += 1
                    else:
                        no_improvement_count = 0
                    last_loss = current_loss
                    
                    # 反向传播
                    accelerator.backward(loss)
                    
                    # 梯度裁剪
                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(
                            params_to_optimize, 1.0
                        )
                            
                    # 优化步骤
                    optimizer.step()
                    optimizer.zero_grad()
                
                # 日志记录和检查点保存
                if global_step % 50 == 0 and accelerator.is_main_process:
                    print(f"\n步骤 {global_step}/{max_train_steps}: 总损失={loss.detach().item():.4f}, "
                        f"实例={instance_loss.detach().item() if isinstance(instance_loss, torch.Tensor) else instance_loss:.4f}, "
                        f"类别={class_loss.detach().item() if isinstance(class_loss, torch.Tensor) else class_loss:.4f}")
                    
                    # 保存检查点
                    if accelerator.is_local_main_process:
                        checkpoint = {
                            "global_step": global_step,
                            "optimizer": optimizer.state_dict(),
                            "unet": accelerator.unwrap_model(unet).state_dict(),
                        }
                        if train_text_encoder:
                            checkpoint["text_encoder"] = accelerator.unwrap_model(text_encoder).state_dict()
                        
                        torch.save(checkpoint, checkpoint_path)
                        print(f"保存检查点到步骤 {global_step}")
                    
                # 更新进度条
                progress_bar.update(1)
                global_step += 1
                
                if global_step >= max_train_steps:
                    break
                
                # 检测是否需要进行早停
                if no_improvement_count >= early_stop_threshold:
                    print(f"\n警告: 连续 {early_stop_threshold} 步损失没有明显改善")
                    print("但训练将继续进行，这可能是优化过程的正常波动")
                    no_improvement_count = 0  # 重置计数器继续训练
        
        # 第9阶段：训练循环结束
        update_stage()
        if HAS_THEORY_NOTES:
            print("\n" + "-"*60)
            print("训练循环完成")
            print("-"*60)
            print("DreamBooth训练已完成所有步骤，模型现在应该已学会了：")
            print(f"1. 将标识符 '{identifier}' 与您特定的主体关联")
            print(f"2. 在保持类别一般知识的同时，记住特定主体的视觉特征")
            print(f"3. 能够根据'{instance_prompt}'生成包含特定主体的图像")

    except KeyboardInterrupt:
        print("\n训练被用户中断")
        print("保存当前模型状态...")
        
        # 保存中断时的检查点
        if accelerator.is_local_main_process:
            checkpoint = {
                "global_step": global_step,
                "optimizer": optimizer.state_dict(),
                "unet": accelerator.unwrap_model(unet).state_dict(),
            }
            if train_text_encoder:
                checkpoint["text_encoder"] = accelerator.unwrap_model(text_encoder).state_dict()
            
            torch.save(checkpoint, os.path.join(output_dir, "interrupt_checkpoint.pt"))
            print(f"中断检查点已保存，您可以稍后恢复训练")
    
    except Exception as e:
        print(f"\n训练遇到错误: {str(e)}")
        import traceback
        traceback.print_exc()
        print("\n尝试保存当前模型...")
        
        if accelerator.is_local_main_process:
            try:
                checkpoint = {
                    "global_step": global_step,
                    "optimizer": optimizer.state_dict(),
                    "unet": accelerator.unwrap_model(unet).state_dict(),
                }
                if train_text_encoder:
                    checkpoint["text_encoder"] = accelerator.unwrap_model(text_encoder).state_dict()
                
                torch.save(checkpoint, os.path.join(output_dir, "error_checkpoint.pt"))
                print("错误检查点已保存")
            except:
                print("保存错误检查点失败")
    
    # 第10阶段：模型保存
    update_stage()
    # 模型保存 - 打印相关理论信息
    if HAS_THEORY_NOTES and accelerator.is_main_process:
        step_info = get_training_step("model_saving")
        if step_info:
            print("\n" + "-"*60)
            print("保存训练成果")
            print("-"*60)
            print(step_info["description"])
    
    # 保存模型
    accelerator.wait_for_everyone()
    
    if accelerator.is_main_process:
        # 解包装模型
        unet = accelerator.unwrap_model(unet)
        text_encoder = accelerator.unwrap_model(text_encoder)
        
        # 保存标识符
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, "identifier.txt"), "w") as f:
            f.write(identifier)
        
        # 尝试保存损失图表
        if len(loss_history["steps"]) > 0:
            try:
                import matplotlib.pyplot as plt
                plt.figure(figsize=(10, 5))
                plt.plot(loss_history["steps"], loss_history["instance"], label="Instance Loss")
                plt.plot(loss_history["steps"], loss_history["class"], label="Class Loss")
                plt.plot(loss_history["steps"], loss_history["total"], label="Total Loss")
                plt.xlabel("Training Steps")
                plt.ylabel("Loss")
                plt.title(f"DreamBooth Training Loss (Completed {global_step}/{max_train_steps} Steps)")
                plt.legend()
                plt.savefig(os.path.join(output_dir, "training_loss.png"))
                print(f"训练损失图表已保存至 {os.path.join(output_dir, 'training_loss.png')}")
            except:
                print("保存损失图表失败，跳过此步骤")
        
        # 保存完整模型
        print("正在保存微调后的模型...")
        pipeline = StableDiffusionPipeline.from_pretrained(
            pretrained_model_name_or_path,
            unet=unet, text_encoder=text_encoder,
            tokenizer=tokenizer, scheduler=noise_scheduler, vae=vae
        )
        pipeline.save_pretrained(output_dir)
        print(f"模型已成功保存到 {output_dir}")
        
        # 更准确地反映训练完成状态
        training_percentage = global_step / max_train_steps * 100
        if training_percentage < 90:
            print(f"\n警告: DreamBooth训练仅完成了 {training_percentage:.1f}% ({global_step}/{max_train_steps}步)")
            print("训练被提前终止。可能的原因:")
            print("1. 内存不足导致程序崩溃")
            print("2. 用户手动中断了训练")
            print("3. 训练过程中出现了错误")
            print("\n虽然模型已保存，但可能无法产生理想效果。建议:")
            print(f"- 使用命令行参数 --resume 从步骤 {global_step} 继续训练")
            print("- 或减少先验图像数量后重新开始训练")
            print(f"- 或减少训练步数到 {global_step} 左右（如果当前结果已经可接受）")
        else:
            print(f"\nDreamBooth训练完成! 共执行了 {global_step}/{max_train_steps} 步 ({training_percentage:.1f}%)")
        
        print(f"使用标识符 '{identifier}' 在提示词中引用您的特定主体")
        
        # 训练完成后打印应用场景理论
        if HAS_THEORY_NOTES:
            theory = get_theory_step("completion")
            if theory:
                print_theory_step("5", theory["title"], theory["description"])
                
            # 打印评估指标理论
            print("\n" + "-"*60)
            print("【评估指标与方法】")
            print("-"*60)
            print(DreamBoothTheory.evaluation_metrics())
    
    # 第11阶段：应用建议
    update_stage()
    # 如果有理论笔记，提供进一步学习和应用的建议
    if HAS_THEORY_NOTES and accelerator.is_main_process:
        print("\n" + "="*60)
        print("【DreamBooth应用指南】")
        print("="*60)
        print(f"""
现在您已成功训练完成DreamBooth模型，可以尝试以下应用场景：

1. 主体重新上下文化 (论文图1):
   - "a {identifier} {class_prompt.replace('a ', '')} in a luxurious living room"
   - "a {identifier} {class_prompt.replace('a ', '')} on a sandy beach"

2. 视角合成 (论文图4):
   - "a front view of {identifier} {class_prompt.replace('a ', '')}"
   - "a side view of {identifier} {class_prompt.replace('a ', '')}"

3. 风格转换 (论文图3):
   - "a painting of {identifier} {class_prompt.replace('a ', '')} in the style of Van Gogh"
   - "a sketch of {identifier} {class_prompt.replace('a ', '')}"

4. 属性编辑 (论文图4):
   - "a {identifier} {class_prompt.replace('a ', '')} wearing a hat"
   - "a {identifier} {class_prompt.replace('a ', '')} made of gold"

使用方式:
python dreambooth.py --infer --model_path {output_dir} --prompt "在此处输入上述示例提示词"
""")
    
    return identifier