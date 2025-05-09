import os
import random
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm
from PIL import Image

# 添加matplotlib用于绘制损失曲线
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from diffusers import StableDiffusionPipeline, DDPMScheduler, UNet2DConditionModel, AutoencoderKL
from diffusers.optimization import get_scheduler
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
from db_modules.training_loop import execute_training_loop
from db_modules.prior_generation import generate_prior_images, check_prior_images
from db_modules.model_saving import save_trained_model
from db_modules.loss_monitor import LossMonitor, verify_instance_images, adjust_training_params

class DreamBoothDataset(Dataset):
    def __init__(self, instance_images_path, class_images_path, tokenizer, size, center_crop, instance_prompt, class_prompt):
        self.size = size
        self.center_crop = center_crop
        self.tokenizer = tokenizer
        self.instance_prompt = instance_prompt
        self.class_prompt = class_prompt

        # ---- START DATASET DEBUG ----
        print(f"[DreamBoothDataset DEBUG] Initializing dataset.")
        print(f"[DreamBoothDataset DEBUG] Received instance_images_path: '{instance_images_path}'")
        
        if not os.path.exists(instance_images_path):
            print(f"[DreamBoothDataset DEBUG WARNING] Instance images path does NOT exist: '{instance_images_path}'")
            self.instance_images = []
        else:
            print(f"[DreamBoothDataset DEBUG] Instance images path exists. Attempting to load images...")
            self.instance_images = self._load_images(instance_images_path)
            print(f"[DreamBoothDataset DEBUG] Loaded {len(self.instance_images)} instance images.")
            if not self.instance_images:
                print(f"[DreamBoothDataset DEBUG WARNING] No instance images were loaded from '{instance_images_path}'. Check the path and image files within.")

        print(f"[DreamBoothDataset DEBUG] Received class_images_path: '{class_images_path}'")
        if class_images_path and os.path.exists(class_images_path):
            print(f"[DreamBoothDataset DEBUG] Class images path exists. Attempting to load class images...")
            self.class_images = self._load_images(class_images_path)
            print(f"[DreamBoothDataset DEBUG] Loaded {len(self.class_images)} class images.")
        elif class_images_path:
            print(f"[DreamBoothDataset DEBUG WARNING] Class images path was provided but does NOT exist: '{class_images_path}'")
            self.class_images = []
        else:
            print(f"[DreamBoothDataset DEBUG] No class images path provided or path is empty.")
            self.class_images = []
        
        print(f"[DreamBoothDataset DEBUG] Total dataset length (instance + class images): {len(self)}")
        if len(self.instance_images) == 0 and len(self.class_images) > 0:
            print(f"[DreamBoothDataset DEBUG CRITICAL WARNING] No instance images loaded, but class images are present. All batches will lack instance data, leading to instance_loss = 0.")
        elif len(self.instance_images) == 0 and len(self.class_images) == 0:
            print(f"[DreamBoothDataset DEBUG CRITICAL WARNING] No instance OR class images loaded. Dataset is empty!")
        
        # 添加这个标志来跟踪最后一个返回的是哪种类型的图片
        self.last_was_instance = False
        
        # 计算实例图片和类别图片的混合比例
        total_images = len(self.instance_images) + len(self.class_images)
        if len(self.instance_images) > 0:
            # 至少确保实例图片占比不少于20%
            self.min_instance_ratio = max(0.2, len(self.instance_images) / total_images)
            print(f"[DreamBoothDataset DEBUG] 设置最小实例图片比例为: {self.min_instance_ratio:.2f}")
        else:
            self.min_instance_ratio = 0.0
        # ---- END DATASET DEBUG ----

    def _load_images(self, path):
        images = []
        valid_extensions = ('.png', '.jpg', '.jpeg', '.webp', '.bmp', '.gif')
        print(f"[DreamBoothDataset._load_images DEBUG] Loading images from path: {path}")
        if not os.path.isdir(path):
            print(f"[DreamBoothDataset._load_images DEBUG WARNING] Path is not a directory: {path}")
            return images
            
        image_files_found = 0
        for img_file in os.listdir(path):
            if img_file.lower().endswith(valid_extensions):
                image_files_found += 1
                try:
                    img_path = os.path.join(path, img_file)
                    img = Image.open(img_path).convert("RGB")
                    
                    if self.center_crop:
                        width, height = img.size
                        if width != height:
                            short, long = (width, height) if width < height else (height, width)
                            left = (width - short) // 2
                            top = (height - short) // 2
                            right = left + short
                            bottom = top + short
                            img = img.crop((left, top, right, bottom))
                    img = img.resize((self.size, self.size), Image.LANCZOS)
                    images.append(img)
                except Exception as e:
                    print(f"[DreamBoothDataset._load_images WARNING] Failed to load or process image {img_file} from {path}: {e}")
        print(f"[DreamBoothDataset._load_images DEBUG] Found {image_files_found} potential image files, successfully loaded {len(images)} images from {path}.")
        return images

    def __len__(self):
        return len(self.instance_images) + len(self.class_images)

    def __getitem__(self, idx):
        
        # 交替返回实例和类别图片的策略，确保实例图片得到足够训练
        if len(self.instance_images) > 0 and (
            self.last_was_instance == False or    # 如果上次返回的是类别图片
            idx % 5 == 0 or                      # 或者按照一定间隔强制返回实例图片
            random.random() < self.min_instance_ratio  # 或者随机概率返回实例图片
        ):
            image = self.instance_images[idx % len(self.instance_images)]
            is_instance_current = True
            self.last_was_instance = True
        elif idx < len(self.instance_images):
            image = self.instance_images[idx]
            is_instance_current = True
            self.last_was_instance = True
        elif idx < len(self.instance_images) + len(self.class_images):
            image = self.class_images[idx - len(self.instance_images)]
            is_instance_current = False
            self.last_was_instance = False
        else:
            # This case should ideally not be hit if __len__ is correct and DataLoader respects it.
            print(f"[DreamBoothDataset DEBUG CRITICAL] Index {idx} is out of bounds. Dataset length: {len(self)}")
            raise IndexError(f"Index {idx} out of bounds for dataset with length {len(self)}")

        image_type_str = "实例 (instance)" if is_instance_current else "类别 (class)"
        print(f"正在加载图片 {idx + 1}/{len(self)} (类型: {image_type_str}) 用于准备训练批次。")

        image_np = np.array(image).astype(np.float32)
        image_tensor = torch.from_numpy(image_np / 127.5 - 1.0)
        image_tensor = image_tensor.permute(2, 0, 1)
        
        return {"pixel_values": image_tensor, "is_instance": torch.tensor(is_instance_current, dtype=torch.bool)}

def load_config(config_path):
    import json
    with open(config_path, "r") as f:
        return json.load(f)

def plot_loss_curves(loss_history, output_dir, prefix=""):
    """
    绘制训练过程中的损失曲线并保存
    
    Args:
        loss_history (dict): 包含'instance', 'class', 'total', 'steps'键的字典
        output_dir (str): 保存图表的目录
        prefix (str): 文件名前缀，可用于区分不同的训练运行
    """
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 跳过空的损失历史
    if not loss_history["steps"]:
        print("损失历史为空，无法绘制损失曲线。")
        return
    
    # 创建一个新的图表
    plt.figure(figsize=(12, 8))
    
    # 绘制总损失
    if loss_history["total"]:
        plt.plot(loss_history["steps"], loss_history["total"], 'b-', 
                 linewidth=2, label='总损失')
    
    # 绘制实例损失 (如果有)
    if any(x != 0 for x in loss_history["instance"]):
        plt.plot(loss_history["steps"], loss_history["instance"], 'g-', 
                 linewidth=2, label='实例损失')
    
    # 绘制类别损失 (如果有)
    if any(x != 0 for x in loss_history["class"]):
        plt.plot(loss_history["steps"], loss_history["class"], 'r-',
                 linewidth=2, label='类别损失')
    
    # 添加标题和标签
    plt.title('DreamBooth 训练损失', fontsize=16)
    plt.xlabel('训练步骤', fontsize=14)
    plt.ylabel('损失值', fontsize=14)
    plt.ylabel('损失值', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)
    
    # 强制x轴显示整数刻度(训练步骤)
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    
    # 如果损失有剧烈波动，可以考虑使用对数尺度
    if max(loss_history["total"]) / (min(x for x in loss_history["total"] if x > 0) + 1e-8) > 100:
        plt.yscale('log')
        plt.title('DreamBooth 训练损失 (对数尺度)', fontsize=16)
    
    # 保存图表
    save_path = os.path.join(output_dir, f"{prefix}loss_curve.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"损失曲线已保存至: {save_path}")
    
    # 保存损失数据为CSV，便于后续分析
    import csv
    csv_path = os.path.join(output_dir, f"{prefix}loss_data.csv")
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Step', 'Total Loss', 'Instance Loss', 'Class Loss'])
        for i, step in enumerate(loss_history["steps"]):
            writer.writerow([
                step, 
                loss_history["total"][i], 
                loss_history["instance"][i], 
                loss_history["class"][i]
            ])
    print(f"损失数据已保存至: {csv_path}")

def load_loss_history(csv_path):
    """
    从CSV文件加载损失历史。
    
    Args:
        csv_path (str): CSV文件路径。

    Returns:
        dict: 包含'instance', 'class', 'total', 'steps'键的字典。
    """
    import csv
    loss_history = {"steps": [], "total": [], "instance": [], "class": []}
    if not os.path.exists(csv_path):
        return loss_history

    with open(csv_path, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            loss_history["steps"].append(int(row["Step"]))
            loss_history["total"].append(float(row["Total Loss"]))
            loss_history["instance"].append(float(row["Instance Loss"]))
            loss_history["class"].append(float(row["Class Loss"]))
    return loss_history

def append_loss_to_csv(csv_path, step, total_loss, instance_loss, class_loss):
    """
    将损失值追加到CSV文件。

    Args:
        csv_path (str): CSV文件路径。
        step (int): 当前训练步骤。
        total_loss (float): 总损失。
        instance_loss (float): 实例损失。
        class_loss (float): 类别损失。
    """
    import csv
    file_exists = os.path.isfile(csv_path)
    with open(csv_path, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        if not file_exists:
            writer.writerow(['Step', 'Total Loss', 'Instance Loss', 'Class Loss'])
        writer.writerow([step, total_loss, instance_loss, class_loss])

def dreambooth_training(config):
    """DreamBooth核心训练逻辑"""

    # 在创建 Accelerator 之前，确保日志目录正确设置
    # 检查当前 Accelerator 版本是否支持 logging_dir 参数
    accelerator_kwargs = {
        "gradient_accumulation_steps": config["training"]["gradient_accumulation_steps"],
        "mixed_precision": config["training"]["mixed_precision"],
    }
    
    # 设置 TensorBoard 的日志目录（如果需要）
    if config["logging_saving"].get("report_to") == "tensorboard":
        # 获取日志目录
        logging_dir = config["logging_saving"].get("logging_dir")
        if not logging_dir:
            logging_dir = os.path.join(config["paths"]["output_dir"], "logs")
            print(f"自动设置 logging_dir 为 '{logging_dir}'")
            
        # 确保日志目录存在
        os.makedirs(logging_dir, exist_ok=True)
        
        # 设置环境变量，让 TensorBoard 知道在哪里保存日志
        os.environ["TENSORBOARD_LOGDIR"] = logging_dir
        
        # 添加 log_with 参数
        accelerator_kwargs["log_with"] = config["logging_saving"]["report_to"]
        
        # 直接添加 logging_dir 参数
        accelerator_kwargs["logging_dir"] = logging_dir
    
    # 创建 Accelerator 实例
    try:
        accelerator = Accelerator(**accelerator_kwargs)
        print(f"成功创建 Accelerator，参数: {accelerator_kwargs}")
    except TypeError as e:
        print(f"创建 Accelerator 时出现错误：{e}")
        print("尝试使用兼容的参数重新创建...")
        
        # 移除可能不兼容的参数
        if "log_with" in accelerator_kwargs:
            del accelerator_kwargs["log_with"]
            print("删除了 log_with 参数")
        
        # 确保没有使用 TensorBoard 时，不传递 logging_dir 参数
        if "logging_dir" in accelerator_kwargs:
            del accelerator_kwargs["logging_dir"]
            print("删除了 logging_dir 参数")
        
        accelerator = Accelerator(**accelerator_kwargs)
        print(f"成功使用兼容参数创建 Accelerator: {accelerator_kwargs}")
    
    if config["training"]["seed"] is not None:
        torch.manual_seed(config["training"]["seed"])
        random.seed(config["training"]["seed"])
        np.random.seed(config["training"]["seed"])
        if accelerator.is_main_process:
            accelerator.print(f"Set random seed to: {config['training']['seed']}")

    if config["training"]["allow_tf32"]:
        torch.backends.cuda.matmul.allow_tf32 = True
        if accelerator.is_main_process:
            accelerator.print("TF32 on matmul enabled for Ampere GPUs.")

    revision_from_config = config["paths"].get("model_revision", None)

    if accelerator.is_main_process:
        print(f"Loading tokenizer from: {config['paths']['pretrained_model_name_or_path']}")
    tokenizer = CLIPTokenizer.from_pretrained(
        config["paths"]["pretrained_model_name_or_path"], subfolder="tokenizer", revision=revision_from_config
    )
    if accelerator.is_main_process:
        print(f"Loading text_encoder from: {config['paths']['pretrained_model_name_or_path']}")
    text_encoder = CLIPTextModel.from_pretrained(
        config["paths"]["pretrained_model_name_or_path"], subfolder="text_encoder", revision=revision_from_config
    )
    if accelerator.is_main_process:
        print(f"Loading vae from: {config['paths']['pretrained_model_name_or_path']}")
    vae = AutoencoderKL.from_pretrained(
        config["paths"]["pretrained_model_name_or_path"], subfolder="vae", revision=revision_from_config
    )
    if accelerator.is_main_process:
        print(f"Loading unet from: {config['paths']['pretrained_model_name_or_path']}")
    unet = UNet2DConditionModel.from_pretrained(
        config["paths"]["pretrained_model_name_or_path"], subfolder="unet", revision=revision_from_config
    )

    # 增加对SDXL的支持 - 加载第二个文本编码器
    text_encoder_2 = None
    if config.get("advanced", {}).get("model_type", "").lower() == "sdxl":
        try:
            accelerator.print("检测到SDXL配置，正在加载第二个文本编码器...")
            from transformers import CLIPTextModelWithProjection
            text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(
                config["paths"]["pretrained_model_name_or_path"], 
                subfolder="text_encoder_2", 
                revision=revision_from_config
            )
            
            # 应用与第一个编码器相同的训练配置
            if not config["training"]["train_text_encoder"]:
                text_encoder_2.requires_grad_(False)
                text_encoder_2.to(accelerator.device, dtype=torch.float32)
            elif config["training"]["gradient_checkpointing"]:
                text_encoder_2.gradient_checkpointing_enable()
                
            accelerator.print("成功加载SDXL第二个文本编码器")
            
            # 将SDXL第二个文本编码器添加到config供后续使用
            config["text_encoder_2"] = text_encoder_2
        except Exception as e:
            accelerator.print(f"加载SDXL第二个文本编码器失败: {e}")
            accelerator.print("将尝试继续训练，但可能影响结果质量")

    # 确保所有模型在同一设备上（统一设为CUDA）
    device = accelerator.device
    
    # 明确将所有模型移动到目标设备
    vae = vae.to(device)
    text_encoder = text_encoder.to(device)
    unet = unet.to(device)
    if text_encoder_2 is not None:
        text_encoder_2 = text_encoder_2.to(device)

    # Freeze VAE
    vae.requires_grad_(False)
    
    # 验证模型已经正确移动到设备上
    if accelerator.is_main_process:
        model_devices = {
            "VAE": next(vae.parameters()).device,
            "Text Encoder": next(text_encoder.parameters()).device,
            "UNet": next(unet.parameters()).device
        }
        if text_encoder_2 is not None:
            model_devices["Text Encoder 2"] = next(text_encoder_2.parameters()).device
            
        accelerator.print("[设备验证] 各模型当前设备:")
        for model_name, model_device in model_devices.items():
            accelerator.print(f"  - {model_name}: {model_device}")
            if model_device != device:
                accelerator.print(f"    [警告] {model_name}不在目标设备({device})上!")
                # 再次尝试移动
                if model_name == "VAE":
                    vae = vae.to(device)
                elif model_name == "Text Encoder":
                    text_encoder = text_encoder.to(device)
                elif model_name == "UNet":
                    unet = unet.to(device)
                elif model_name == "Text Encoder 2":
                    text_encoder_2 = text_encoder_2.to(device)

    # Handle Text Encoder:
    if not config["training"]["train_text_encoder"]:
        text_encoder.requires_grad_(False)
        text_encoder.to(accelerator.device, dtype=torch.float32)
    # else: text_encoder will be handled by accelerator.prepare and its requires_grad should be True by default

    # UNet's gradient_checkpointing is enabled here if configured, before prepare
    if config["training"]["gradient_checkpointing"]:
        unet.enable_gradient_checkpointing()
        if config["training"]["train_text_encoder"]: # Only if text_encoder is also meant to be trained
            text_encoder.gradient_checkpointing_enable()

    # 添加显存优化配置
    # 判断是否需要进行特殊的显存优化
    if config["memory_optimization"].get("low_memory_mode", False):
        accelerator.print("启用低显存训练模式")
        
        # 应用额外优化
        try:
            # 尝试导入并应用优化
            from db_modules.memory_optimization import optimize_model_for_training
            
            mem_config = {
                "attention_slice_size": config["memory_optimization"].get("attention_slice_size", 4),
                "gradient_checkpointing": config["training"].get("gradient_checkpointing", True),
                "disable_text_encoder_training": not config["training"].get("train_text_encoder", False),
                "xformers_optimization": config["memory_optimization"].get("xformers_optimization", True)
            }
            
            unet, text_encoder = optimize_model_for_training(unet, text_encoder, mem_config)
            accelerator.print("已应用内存优化设置")
            
            # 如果是SDXL，同样优化第二个文本编码器
            if text_encoder_2 is not None and config["training"]["train_text_encoder"]:
                text_encoder_2 = optimize_model_for_training(text_encoder_2, None, mem_config)[0]
        except ImportError:
            accelerator.print("无法导入内存优化模块，跳过高级内存优化")
        except Exception as e:
            accelerator.print(f"应用内存优化时出错: {e}")

    # Optimizer
    params_to_optimize = list(filter(lambda p: p.requires_grad, unet.parameters()))
    if config["training"]["train_text_encoder"]:
        params_to_optimize.extend(list(filter(lambda p: p.requires_grad, text_encoder.parameters())))
    
    if not params_to_optimize:
        accelerator.print("CRITICAL ERROR: No parameters found to optimize. \n"
                          "This usually means that the UNet (and Text Encoder, if 'train_text_encoder' is true) "
                          "has no layers with 'requires_grad=True'. \n"
                          "Please check your model loading, 'train_text_encoder' setting in config.json, "
                          "and ensure that 'requires_grad_' has not been inadvertently set to False on all trainable layers. "
                          "Training cannot proceed.")
        # Determine identifier for return
        prompt_parts = config["dataset"]["instance_prompt"].split(" ")
        identifier_token = "initialization_error" # Default error identifier
        if len(prompt_parts) >= 2 and prompt_parts[-2].lower() not in ["a", "an", "of", "the"]: # Basic heuristic
            identifier_token = prompt_parts[-2]
        return identifier_token, False # Indicate failure

    optimizer = None # Initialize to None
    if config["memory_optimization"]["use_8bit_adam"]:
        try:
            import bitsandbytes as bnb
            optimizer = bnb.optim.AdamW8bit(
                params_to_optimize,
                lr=config["training"]["learning_rate"],
                betas=(config["training"]["adam_beta1"], config["training"]["adam_beta2"]),
                weight_decay=config["training"]["adam_weight_decay"],
                eps=config["training"]["adam_epsilon"],
            )
            accelerator.print("Using 8-bit AdamW optimizer.")
        except ImportError:
            accelerator.print("bitsandbytes not found or import failed. Using standard AdamW optimizer.")
            optimizer = torch.optim.AdamW( # Ensure this assignment happens
                params_to_optimize,
                lr=config["training"]["learning_rate"],
                betas=(config["training"]["adam_beta1"], config["training"]["adam_beta2"]),
                weight_decay=config["training"]["adam_weight_decay"], # 修正参数名称
                eps=config["training"]["adam_epsilon"],
            )
        except Exception as e: # Catch any other exception during 8bit Adam creation
            accelerator.print(f"Error creating 8-bit AdamW optimizer: {e}. Using standard AdamW optimizer.")
            optimizer = torch.optim.AdamW(
                params_to_optimize,
                lr=config["training"]["learning_rate"],
                betas=(config["training"]["adam_beta1"], config["training"]["adam_beta2"]),
                weight_decay=config["training"]["adam_weight_decay"], # 修正参数名称
                eps=config["training"]["adam_epsilon"],
            )
    else:
        optimizer = torch.optim.AdamW(
            params_to_optimize,
            lr=config["training"]["learning_rate"],
            betas=(config["training"]["adam_beta1"], config["training"]["adam_beta2"]),
            weight_decay=config["training"]["adam_weight_decay"], # 修正参数名称
            eps=config["training"]["adam_epsilon"],
        )
        accelerator.print("Using standard AdamW optimizer.")

    if optimizer is None:
        accelerator.print("CRITICAL ERROR: Optimizer was not created. Training cannot proceed.")
        # Determine identifier for return
        prompt_parts = config["dataset"]["instance_prompt"].split(" ")
        identifier_token = "optimizer_creation_error" 
        if len(prompt_parts) >= 2 and prompt_parts[-2].lower() not in ["a", "an", "of", "the"]:
            identifier_token = prompt_parts[-2]
        return identifier_token, False

    # LR Scheduler
    lr_scheduler = get_scheduler(
        config["training"]["lr_scheduler"],
        optimizer=optimizer,
        num_warmup_steps=config["training"]["lr_warmup_steps"] * config["training"]["gradient_accumulation_steps"],
        num_training_steps=config["training"]["max_train_steps"] * config["training"]["gradient_accumulation_steps"],
    )

    train_dataset = DreamBoothDataset(
        instance_images_path=config["paths"]["instance_data_dir"],
        class_images_path=config["paths"]["class_data_dir"],
        tokenizer=tokenizer,
        size=config["dataset"]["resolution"],
        center_crop=config["dataset"]["center_crop"],
        instance_prompt=config["dataset"]["instance_prompt"],
        class_prompt=config["dataset"]["class_prompt"]
    )

    if len(train_dataset) == 0:
        accelerator.print("CRITICAL: Dataset is empty. Please check instance and class image paths and content.")
        return config["dataset"]["instance_prompt"], False # identifier, success

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config["dataset"]["train_batch_size"], # Changed from config["training"]
        shuffle=True,
        num_workers=config["dataset"]["dataloader_num_workers"],
        collate_fn=None, # Add custom collate_fn if needed
    )

    # 确认数据加载器批次大小匹配预期
    if len(train_dataloader) == 0:
        batch_size_warning = (
            f"警告: 数据加载器为空! 批次大小({config['dataset']['train_batch_size']})可能大于数据集大小({len(train_dataset)})"
        )
        accelerator.print(batch_size_warning)

    # Determine the mixed_precision_dtype that accelerator is using for its models
    actual_mixed_precision_dtype = torch.float32 # Default if "no" or unrecognized
    if config["training"]["mixed_precision"] == "fp16":
        actual_mixed_precision_dtype = torch.float16
    elif config["training"]["mixed_precision"] == "bf16":
        actual_mixed_precision_dtype = torch.bfloat16
    
    accelerator.print(f"Using mixed precision: {config['training']['mixed_precision']}, corresponding to torch.dtype: {actual_mixed_precision_dtype}")

    # 加载损失历史之前，确保类别图像目录存在
    class_images_path = config["paths"]["class_data_dir"]
    if class_images_path:
        os.makedirs(class_images_path, exist_ok=True)
        accelerator.print(f"已确保类别图像目录存在: {class_images_path}")
    
    # 加载损失历史
    loss_csv_path = os.path.join(config["paths"]["output_dir"], "loss_data.csv")
    loss_history = load_loss_history(loss_csv_path)
    
    # 确保输出目录存在
    os.makedirs(config["paths"]["output_dir"], exist_ok=True)
    
    # 改进的断点重训逻辑
    # 获取配置中的检查点路径或使用默认路径
    checkpoint_dir = config["logging_saving"].get("save_model_config", {}).get("checkpoint_path", config["paths"]["output_dir"])
    # 确保检查点目录存在
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # 使用配置中的中断检查点路径或构建默认路径
    interrupt_checkpoint_path = config["logging_saving"].get("save_model_config", {}).get("interrupt_checkpoint_path", 
                                os.path.join(config["paths"]["output_dir"], "interrupt_checkpoint.pt"))
    
    accelerator.print(f"使用中断检查点路径: {interrupt_checkpoint_path}")
    resume_step = 0
    
    # 先检查是否有之前的训练损失记录
    if loss_history["steps"]:
        last_recorded_step = loss_history["steps"][-1]
        print(f"找到之前的训练记录，最后记录的步骤为: {last_recorded_step}")
        resume_step = last_recorded_step + 1
    
    # 检查checkpoint文件是否存在
    if os.path.exists(interrupt_checkpoint_path):
        print(f"找到断点重训检查点: {interrupt_checkpoint_path}")
        try:
            checkpoint = torch.load(interrupt_checkpoint_path, map_location="cpu")
            checkpoint_step = checkpoint.get("global_step", 0)
            
            # 如果检查点步骤大于loss记录的步骤，则使用检查点步骤
            if checkpoint_step > resume_step:
                resume_step = checkpoint_step
                print(f"基于检查点文件更新恢复步骤为: {resume_step}")
                
                # 加载优化器状态
                if "optimizer" in checkpoint:
                    optimizer.load_state_dict(checkpoint["optimizer"])
                    print("成功加载优化器状态")
                
                # 加载UNet权重
                if "unet" in checkpoint:
                    unet.load_state_dict(checkpoint["unet"])
                    print("成功加载UNet权重")
                
                # 如果需要，加载文本编码器权重
                if config["training"]["train_text_encoder"] and "text_encoder" in checkpoint:
                    text_encoder.load_state_dict(checkpoint["text_encoder"])
                    print("成功加载Text Encoder权重")
            else:
                print(f"检查点步骤({checkpoint_step})不大于当前记录的损失步骤({resume_step-1})，使用损失历史的恢复点")
        except Exception as e:
            print(f"加载检查点时出错: {e}")
            print("将从头开始训练或使用损失记录的最后步骤")
    
    # 检查resume_from_checkpoint配置
    if config["training"].get("resume_from_checkpoint"):
        custom_checkpoint = config["training"]["resume_from_checkpoint"]
        if os.path.exists(custom_checkpoint):
            print(f"使用自定义检查点: {custom_checkpoint}")
            try:
                checkpoint = torch.load(custom_checkpoint, map_location="cpu")
                checkpoint_step = checkpoint.get("global_step", 0)
                resume_step = checkpoint_step
                
                # 加载模型和优化器状态
                if "optimizer" in checkpoint:
                    optimizer.load_state_dict(checkpoint["optimizer"])
                if "unet" in checkpoint:
                    unet.load_state_dict(checkpoint["unet"])
                if config["training"]["train_text_encoder"] and "text_encoder" in checkpoint:
                    text_encoder.load_state_dict(checkpoint["text_encoder"])
                
                print(f"成功从自定义检查点恢复训练，步骤: {resume_step}")
            except Exception as e:
                print(f"加载自定义检查点时出错: {e}")
    
    print(f"最终确定的恢复训练步骤: {resume_step}")
    
    # 如果步骤大于等于最大步数，提醒用户
    if resume_step >= config["training"]["max_train_steps"]:
        print(f"警告: 恢复步骤({resume_step})已达到或超过最大训练步骤({config['training']['max_train_steps']})。")
        print("您可能需要增加max_train_steps或从头开始训练。")
    
    global_step, loss_history, training_successful = execute_training_loop(
        accelerator=accelerator,
        unet=unet, 
        text_encoder=text_encoder, 
        vae=vae,
        tokenizer=tokenizer,
        dataloader=train_dataloader,
        optimizer=optimizer,
        noise_scheduler=lr_scheduler,
        lr_scheduler=lr_scheduler,
        config=config, 
        resume_step=resume_step,  # 传递正确的恢复步骤
        memory_mgr=None,
        debug_monitor=None,
        loss_monitor=None,
        mixed_precision_dtype=actual_mixed_precision_dtype,
        loss_csv_path=loss_csv_path,  # 传递CSV路径
        text_encoder_2=text_encoder_2  # 传递第二个文本编码器
    )

    if accelerator.is_main_process:
        # 在训练结束后绘制损失曲线
        if loss_history and len(loss_history["steps"]) > 0:
            prefix = f"{config['dataset']['instance_prompt'].split()[-1]}_"
            plot_loss_curves(loss_history, config["paths"]["output_dir"], prefix)
            
        accelerator.end_training()

    prompt_parts = config["dataset"]["instance_prompt"].split(" ")
    identifier_token = "sks"
    if len(prompt_parts) >= 2 and prompt_parts[-2] != "a" and prompt_parts[-2] != "of":
        identifier_token = prompt_parts[-2]
    
    return identifier_token, training_successful

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_file",
        type=str,
        default="config.json",
        help="Path to the training configuration JSON file.",
    )
    args = parser.parse_args()

    if not os.path.exists(args.config_file):
        print(f"Error: Config file '{args.config_file}' not found.")
        exit(1)
        
    config_data = load_config(args.config_file)

    identifier, success = dreambooth_training(config_data)
    if success:
        print(f"DreamBooth training completed successfully for identifier: {identifier}")
    else:
        print(f"DreamBooth training failed or was interrupted for identifier: {identifier}.")