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

# 导入调试工具
try:
    from debug_tools import DebugMonitor
    HAS_DEBUG_TOOLS = True
except ImportError:
    HAS_DEBUG_TOOLS = False
    print("未找到调试工具模块，将不会生成详细的训练日志")

# 导入拆分的功能模块
try:
    from db_modules.training_loop import execute_training_loop
    from db_modules.prior_generation import generate_prior_images, check_prior_images
    from db_modules.model_saving import save_trained_model
    MODULES_AVAILABLE = True
except ImportError:
    MODULES_AVAILABLE = False
    print("未找到拆分的功能模块，将使用内置实现")

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
    resume_training=False,
    # 新增低内存优化参数
    attention_slice_size=0,
    gradient_checkpointing=False,
    use_8bit_adam=False,
):
    """DreamBooth核心训练逻辑"""
    # 初始化调试监控器
    debug_monitor = None
    if HAS_DEBUG_TOOLS:
        debug_monitor = DebugMonitor(output_dir)
    
    # 分阶段执行训练流程
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
    
    # 应用低内存优化
    if attention_slice_size > 0:
        unet.set_attention_slice(attention_slice_size)
        print(f"已启用注意力切片，大小: {attention_slice_size}")
    
    if gradient_checkpointing:
        unet.enable_gradient_checkpointing()
        if train_text_encoder:
            text_encoder.gradient_checkpointing_enable()
        print("已启用梯度检查点以降低内存使用")
    
    vae.requires_grad_(False)
    vae.to(accelerator.device, dtype=torch.float32)
    
    # 第3阶段：标识符选择
    update_stage()
    identifier = find_rare_token(tokenizer)
    
    if instance_prompt is None:
        class_name = class_prompt.replace("a ", "").strip()
        instance_prompt = f"a {identifier} {class_name}"
    
    print(f"实例提示词: '{instance_prompt}'")
    print(f"类别提示词: '{class_prompt}'")
    
    # 第4阶段：先验图像生成
    update_stage()
    
    pipeline = StableDiffusionPipeline.from_pretrained(
        pretrained_model_name_or_path,
        vae=vae, text_encoder=text_encoder, unet=unet,
    )
    pipeline.to(accelerator.device)
    
    if MODULES_AVAILABLE:
        class_images_dir = generate_prior_images(
            pipeline=pipeline,
            class_prompt=class_prompt,
            output_dir=output_dir,
            num_samples=prior_generation_samples,
            theory_notes_enabled=HAS_THEORY_NOTES,
            theory_step_fn=get_theory_step
        )
    else:
        class_images_dir = generate_class_images(pipeline, class_prompt, os.path.join(output_dir, "class_images"), prior_generation_samples)
    
    del pipeline
    if memory_mgr:
        memory_mgr.cleanup("释放生成管道后")
    
    # 第5阶段：数据集构建
    update_stage()
    instance_images_count = len([f for f in os.listdir(instance_data_dir) 
                              if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    if instance_images_count == 0:
        raise ValueError(f"错误: 在'{instance_data_dir}'目录中未找到任何图像文件")
    elif instance_images_count < 3:
        print(f"警告: 仅发现{instance_images_count}张实例图像。DreamBooth建议使用3-5张图像以获得最佳效果")
    else:
        print(f"已发现{instance_images_count}张实例图像，符合DreamBooth推荐的3-5张图像范围")
    
    dataset = DreamBoothDataset(
        instance_images_path=instance_data_dir,
        class_images_path=class_images_dir,
        tokenizer=tokenizer,
    )
    print(f"创建了训练数据集，包含{len(dataset.instance_images)}张实例图像和{len(dataset.class_images)}张类别图像")
    
    dataloader = DataLoader(dataset, batch_size=train_batch_size, shuffle=True)
    
    # 第6阶段：优化器配置
    update_stage()
    params_to_optimize = (
        list(unet.parameters()) + list(text_encoder.parameters()) 
        if train_text_encoder else unet.parameters()
    )
    
    if use_8bit_adam:
        try:
            import bitsandbytes as bnb
            optimizer = bnb.optim.AdamW8bit(
                params_to_optimize,
                lr=learning_rate,
                betas=(0.9, 0.999),
                weight_decay=1e-2,
            )
            print("使用8位精度Adam优化器以节省内存")
        except ImportError:
            print("未找到bitsandbytes库，回退到标准Adam优化器")
            optimizer = torch.optim.AdamW(
                params_to_optimize,
                lr=learning_rate,
                betas=(0.9, 0.999),
                weight_decay=1e-2,
            )
    else:
        optimizer = torch.optim.AdamW(
            params_to_optimize,
            lr=learning_rate,
            betas=(0.9, 0.999),
            weight_decay=1e-2,
        )
    
    noise_scheduler = DDPMScheduler.from_pretrained(
        pretrained_model_name_or_path, subfolder="scheduler"
    )
    
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
    
    unet, text_encoder, optimizer, dataloader = accelerator.prepare(
        unet, text_encoder, optimizer, dataloader
    )
    
    checkpoint_path = os.path.join(output_dir, "checkpoint.pt")
    start_step = 0
    
    if resume_training and os.path.exists(checkpoint_path):
        print(f"正在从检查点恢复训练...")
        try:
            checkpoint = torch.load(checkpoint_path)
            global_step = checkpoint["global_step"]
            start_step = global_step
            optimizer.load_state_dict(checkpoint["optimizer"])
            unet.load_state_dict(checkpoint["unet"])
            if train_text_encoder and "text_encoder" in checkpoint:
                text_encoder.load_state_dict(checkpoint["text_encoder"])
            print(f"成功恢复训练，从步骤 {global_step} 开始")
        except Exception as e:
            print(f"恢复训练失败: {str(e)}")
            print("将从头开始训练")
    elif os.path.exists(checkpoint_path) and not resume_training:
        print(f"发现检查点文件，但未启用恢复训练。如需恢复训练，请使用 --resume 参数")
    
    # 第7-9阶段：训练循环
    if MODULES_AVAILABLE:
        global_step, loss_history, training_successful = execute_training_loop(
            accelerator=accelerator,
            unet=unet,
            text_encoder=text_encoder,
            vae=vae,
            tokenizer=tokenizer,
            dataloader=dataloader,
            optimizer=optimizer,
            noise_scheduler=noise_scheduler,
            instance_prompt=instance_prompt,
            class_prompt=class_prompt,
            max_train_steps=max_train_steps,
            output_dir=output_dir,
            prior_preservation_weight=prior_preservation_weight,
            gradient_accumulation_steps=gradient_accumulation_steps,
            train_text_encoder=train_text_encoder,
            resume_from=start_step if resume_training else None,
            memory_mgr=memory_mgr,
            debug_monitor=debug_monitor,
            has_theory_notes=HAS_THEORY_NOTES,
            update_stage_fn=update_stage,
            get_theory_step=get_theory_step
        )
    else:
        # 使用内置训练循环
        # ...existing code...
        pass
    
    # 第10阶段：模型保存
    update_stage()
    accelerator.wait_for_everyone()
    
    if accelerator.is_main_process:
        if MODULES_AVAILABLE:
            pipeline = save_trained_model(
                pretrained_model_name_or_path=pretrained_model_name_or_path,
                output_dir=output_dir,
                unet=accelerator.unwrap_model(unet),
                text_encoder=accelerator.unwrap_model(text_encoder),
                tokenizer=tokenizer,
                noise_scheduler=noise_scheduler,
                vae=vae,
                identifier=identifier,
                loss_history=loss_history if 'loss_history' in locals() else None,
                theory_notes_enabled=HAS_THEORY_NOTES
            )
        else:
            # 使用内置模型保存逻辑
            # ...existing code...
            pass
        
    print(f"\nDreamBooth训练完成！使用标识符 '{identifier}' 在提示词中引用您的特定主体")
    
    # 第11阶段：应用建议
    update_stage()
    # ...existing code for application guide...
    
    return identifier