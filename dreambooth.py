import os
import random
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm
# Import Resampling from PIL for the LANCZOS constant
try:
    from PIL import Image, ImageOps
    from PIL.Image import Resampling
    LANCZOS = Resampling.LANCZOS
except ImportError:
    # For older versions of Pillow
    from PIL import Image, ImageOps
    # Fall back to integer constant for very old Pillow versions
    LANCZOS = 1  # This is the integer value that used to represent LANCZOS

# 添加matplotlib用于绘制损失曲线
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# Updated imports to address Pylance warnings
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import StableDiffusionPipeline
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.models.unets.unet_2d_condition import UNet2DConditionModel
from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL
from diffusers.optimization import get_scheduler
from transformers.models.clip import CLIPTextModel, CLIPTokenizer
from accelerate import Accelerator

# 导入理论笔记用于训练过程中打印
try:
    from theory_notes import DreamBoothTheory, get_theory_step, get_training_step, print_theory_step
    HAS_THEORY_NOTES = True
    # print("已加载DreamBooth理论笔记模块，将在训练过程中提供论文解析...") # Quieted
except ImportError:
    HAS_THEORY_NOTES = False
    # print("未找到理论笔记模块，将只显示基本训练信息...") # Quieted

# 导入调试工具
try:
    from debug_tools import DebugMonitor
    HAS_DEBUG_TOOLS = True
except ImportError:
    HAS_DEBUG_TOOLS = False
    # print("未找到调试工具模块，将不会生成详细的训练日志") # Quieted

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

        print(f"[DreamBoothDataset DEBUG] Initializing dataset.")
        print(f"[DreamBoothDataset DEBUG] Received instance_images_path: '{instance_images_path}'")
        
        if not os.path.exists(instance_images_path):
            print(f"[DreamBoothDataset WARNING] Instance images path '{instance_images_path}' does not exist.")
            self.instance_images = []
        else:
            print(f"[DreamBoothDataset DEBUG] Instance images path exists. Attempting to load images...")
            self.instance_images = self._load_images(instance_images_path, "instance")
            if not self.instance_images:
                print(f"[DreamBoothDataset WARNING] No instance images were loaded from '{instance_images_path}'. Check the path and image files within.")
            else:
                print(f"[DreamBoothDataset DEBUG] Loaded {len(self.instance_images)} instance images.")

        print(f"[DreamBoothDataset DEBUG] Received class_images_path: '{class_images_path}'")
        if class_images_path and os.path.exists(class_images_path):
            print(f"[DreamBoothDataset DEBUG] Class images path exists. Attempting to load class images...")
            self.class_images = self._load_images(class_images_path, "class")
            if self.class_images:
                 print(f"[DreamBoothDataset DEBUG] Loaded {len(self.class_images)} class images.")
            else:
                print(f"[DreamBoothDataset WARNING] No class images were loaded from '{class_images_path}'.")
        elif class_images_path:
            print(f"[DreamBoothDataset WARNING] Class images path '{class_images_path}' provided but does not exist.")
            self.class_images = []
        else:
            print(f"[DreamBoothDataset DEBUG] No class_images_path provided.")
            self.class_images = []
        
        if len(self.instance_images) == 0 and len(self.class_images) > 0:
            print(f"[DreamBoothDataset CRITICAL WARNING] No instance images loaded, but class images are present. All batches will lack instance data, leading to instance_loss = 0.")
        elif len(self.instance_images) == 0 and len(self.class_images) == 0:
            print(f"[DreamBoothDataset CRITICAL WARNING] No instance OR class images loaded. Dataset is empty!")
        
        self.last_was_instance = False
        
        total_images = len(self.instance_images) + len(self.class_images)
        if len(self.instance_images) > 0 and total_images > 0:
            self.min_instance_ratio = max(0.2, len(self.instance_images) / total_images)
        else:
            self.min_instance_ratio = 0.0

        print(f"[DreamBoothDataset DEBUG] Total dataset length (instance + class images): {len(self)}")
        print(f"[DreamBoothDataset DEBUG] 设置最小实例图片比例为: {self.min_instance_ratio:.2f}")

    def _load_images(self, path, image_type="unknown"):
        print(f"[DreamBoothDataset._load_images DEBUG] Loading {image_type} images from path: {path}")
        images = []
        valid_extensions = ('.png', '.jpg', '.jpeg', '.webp', '.bmp', '.gif')
        if not os.path.isdir(path):
            print(f"[DreamBoothDataset._load_images WARNING] Path is not a directory: {path}")
            return images
            
        image_files_found = 0
        loaded_image_count = 0
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
                    img = img.resize((self.size, self.size), LANCZOS)
                    images.append(img)
                    loaded_image_count += 1
                except Exception as e:
                    print(f"[DreamBoothDataset._load_images WARNING] Failed to load or process image {img_file} from {path}: {e}")
        print(f"[DreamBoothDataset._load_images DEBUG] Found {image_files_found} potential image files, successfully loaded {loaded_image_count} images from {path}.")
        return images

    def __len__(self):
        return len(self.instance_images) + len(self.class_images)

    def __getitem__(self, idx):
        if len(self.instance_images) > 0 and (
            self.last_was_instance == False or
            idx % 5 == 0 or # Ensure some instance images even if random misses
            random.random() < self.min_instance_ratio
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
            raise IndexError(f"Index {idx} out of bounds for dataset with length {len(self)}")

        image_type_str = "实例 (instance)" if is_instance_current else "类别 (class)"
        # print(f"正在加载图片 {idx + 1}/{len(self)} (类型: {image_type_str}) 用于准备训练批次。") # This can be too verbose, keeping it commented for now.

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
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)
    
    # 强制x轴显示整数刻度(训练步骤)
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    
    # 如果损失有剧烈波动，可以考虑使用对数尺度
    if loss_history["total"] and (min(x for x in loss_history["total"] if x > 0) + 1e-8) > 0 and \
       max(loss_history["total"]) / (min(x for x in loss_history["total"] if x > 0) + 1e-8) > 100:
        plt.yscale('log')
        plt.title('DreamBooth 训练损失 (对数尺度)', fontsize=16)
    
    # 保存图表
    save_path = os.path.join(output_dir, f"{prefix}loss_curve.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"损失曲线已保存至: {save_path}")
    
    # CSV saving is now handled by the training loop via loss_csv_path
    # import csv
    # csv_path = os.path.join(output_dir, f"{prefix}loss_data.csv")
    # with open(csv_path, 'a', newline='') as csvfile:
    #     writer = csv.writer(csvfile)
    #     writer.writerow(['Step', 'Total Loss', 'Instance Loss', 'Class Loss'])
    #     for i in range(len(loss_history['steps'])):
    #         writer.writerow([loss_history['steps'][i], loss_history['total'][i], loss_history['instance'][i], loss_history['class'][i]])
    # print(f"损失数据已保存至: {csv_path}")

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
    except TypeError as e:
        # print(f"创建 Accelerator 时出现错误：{e}") # Quieted
        # print("尝试使用兼容的参数重新创建...") # Quieted
        if "log_with" in accelerator_kwargs:
            del accelerator_kwargs["log_with"]
        if "logging_dir" in accelerator_kwargs: # Specific error from user log
            del accelerator_kwargs["logging_dir"]
        accelerator = Accelerator(**accelerator_kwargs)
    
    target_device = accelerator.device
    accelerator.print(f"Accelerator 主进程: {accelerator.is_main_process}. 使用目标设备: {target_device}")

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
        accelerator.print(f"Loading tokenizer from: {config['paths']['pretrained_model_name_or_path']}")
    tokenizer = CLIPTokenizer.from_pretrained(
        config["paths"]["pretrained_model_name_or_path"], subfolder="tokenizer", revision=revision_from_config
    )
    if accelerator.is_main_process:
        accelerator.print(f"Loading text_encoder from: {config['paths']['pretrained_model_name_or_path']}")
    text_encoder = CLIPTextModel.from_pretrained(
        config["paths"]["pretrained_model_name_or_path"], subfolder="text_encoder", revision=revision_from_config
    ).to(target_device)
    if accelerator.is_main_process:
        accelerator.print(f"Loading vae from: {config['paths']['pretrained_model_name_or_path']}")
    vae = AutoencoderKL.from_pretrained(
        config["paths"]["pretrained_model_name_or_path"], subfolder="vae", revision=revision_from_config
    ).to(target_device)
    if accelerator.is_main_process:
        accelerator.print(f"Loading unet from: {config['paths']['pretrained_model_name_or_path']}")
    unet = UNet2DConditionModel.from_pretrained(
        config["paths"]["pretrained_model_name_or_path"], subfolder="unet", revision=revision_from_config
    ).to(target_device)

    text_encoder_2 = None
    if config.get("advanced", {}).get("model_type", "").lower() == "sdxl":
        try:
            accelerator.print("检测到SDXL配置，正在加载第二个文本编码器...")
            from transformers.models.clip import CLIPTextModelWithProjection
            text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(
                config["paths"]["pretrained_model_name_or_path"], 
                subfolder="text_encoder_2", 
                revision=revision_from_config
            ).to(target_device)
            config["text_encoder_2"] = text_encoder_2 # Store for later use if needed by other parts of config
            accelerator.print("成功加载SDXL第二个文本编码器并移至目标设备")
        except Exception as e:
            accelerator.print(f"加载SDXL第二个文本编码器失败: {e}")

    vae.requires_grad_(False)
    
    if not config["training"]["train_text_encoder"]:
        text_encoder.requires_grad_(False)
        if text_encoder_2:
            text_encoder_2.requires_grad_(False)
    
    if config["training"]["gradient_checkpointing"]:
        unet.enable_gradient_checkpointing()
        if config["training"]["train_text_encoder"]:
            text_encoder.gradient_checkpointing_enable()
            if text_encoder_2:
                 text_encoder_2.gradient_checkpointing_enable()
    
    if accelerator.is_main_process:
        accelerator.print("\n[模型初始设备状态]")
        accelerator.print(f"  - VAE: {next(vae.parameters()).device}")
        accelerator.print(f"  - Text Encoder: {next(text_encoder.parameters()).device}")
        accelerator.print(f"  - UNet: {next(unet.parameters()).device}")
        if text_encoder_2:
            accelerator.print(f"  - Text Encoder 2: {next(text_encoder_2.parameters()).device}")
        accelerator.print(f"预期所有模型均在: {target_device}\n")

    # Memory optimization (models are already on target_device)
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
        if text_encoder_2 and config["training"]["train_text_encoder"]: 
             params_to_optimize.extend(list(filter(lambda p: p.requires_grad, text_encoder_2.parameters())))
    
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
            optimizer_cls = bnb.optim.AdamW8bit
            accelerator.print("Using 8-bit AdamW optimizer.")
        except ImportError:
            accelerator.print("bitsandbytes not found. Falling back to regular AdamW optimizer.")
            optimizer_cls = torch.optim.AdamW
    else:
        optimizer_cls = torch.optim.AdamW

    optimizer = optimizer_cls(
        params_to_optimize,
        lr=config["training"]["learning_rate"],
        betas=(config["training"]["adam_beta1"], config["training"]["adam_beta2"]),
        weight_decay=config["training"]["adam_weight_decay"],
        eps=config["training"]["adam_epsilon"],
    )

    # Dataset and DataLoader
    # ... (dataset initialization, ensure instance_prompt and class_prompt are correctly fetched from config)
    instance_prompt = config["dataset"]["instance_prompt"]
    class_prompt = config["dataset"].get("class_prompt", None)

    train_dataset = DreamBoothDataset(
        instance_images_path=config["paths"]["instance_data_dir"],
        class_images_path=config["paths"].get("class_data_dir"), # Use .get for safety
        tokenizer=tokenizer,
        size=config["dataset"]["resolution"],
        center_crop=config["dataset"]["center_crop"],
        instance_prompt=instance_prompt,
        class_prompt=class_prompt
    )

    train_dataloader = DataLoader(
        train_dataset, batch_size=config["dataset"]["train_batch_size"], shuffle=True, num_workers=config["dataset"]["dataloader_num_workers"]
    )

    # Scheduler and training
    # Initialize noise_scheduler
    noise_scheduler = DDPMScheduler.from_pretrained(
        config["paths"]["pretrained_model_name_or_path"], subfolder="scheduler", revision=revision_from_config
    )

    lr_scheduler = get_scheduler(
        config["training"]["lr_scheduler"],
        optimizer=optimizer,
        num_warmup_steps=config["training"]["lr_warmup_steps"] * config["training"]["gradient_accumulation_steps"],
        num_training_steps=config["training"]["max_train_steps"] * config["training"]["gradient_accumulation_steps"],
    )

    # Initialize LossMonitor and DebugMonitor
    loss_threshold = config.get("training", {}).get("loss_threshold", 1e-5)
    loss_patience = config.get("training", {}).get("loss_patience", 10)
    loss_monitor = LossMonitor(threshold=loss_threshold, patience=loss_patience)
    debug_monitor = None
    if HAS_DEBUG_TOOLS:
        try:
            # Create the DebugMonitor instance with safe parameter passing
            from debug_tools import DebugMonitor
            debug_output_dir = os.path.join(config.get("paths", {}).get("output_dir", "."), "debug_logs")
            log_interval = config.get("logging_saving", {}).get("log_debug_info_every_n_steps", 50)
            os.makedirs(debug_output_dir, exist_ok=True)
            debug_monitor = DebugMonitor(output_dir=debug_output_dir, log_interval=log_interval)
        except Exception as e:
            accelerator.print(f"Failed to initialize DebugMonitor: {e}")
            debug_monitor = None

    # Prepare everything with our 🤗 Accelerate
    # VAE is not part of the optimization loop (vae.requires_grad_(False) is set),
    # but it should still be prepared by Accelerate for consistent device handling.
    if text_encoder_2:
        unet, text_encoder, text_encoder_2, vae, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            unet, text_encoder, text_encoder_2, vae, optimizer, train_dataloader, lr_scheduler
        )
    else:
        unet, text_encoder, vae, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            unet, text_encoder, vae, optimizer, train_dataloader, lr_scheduler
        )
    
    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    # Safely determine max_train_steps
    current_max_train_steps = config["training"].get("max_train_steps", 0)
    if not isinstance(current_max_train_steps, int) or current_max_train_steps <= 0:
        # max_train_steps not provided or invalid, calculate from epochs
        num_train_epochs = config["training"].get("max_train_epochs", 1) # Default to 1 epoch if not specified
        if not isinstance(num_train_epochs, int) or num_train_epochs <= 0: # Ensure epochs is a positive integer
            accelerator.print(f"[WARNING] 'max_train_epochs' is invalid ('{num_train_epochs}'), defaulting to 1 epoch.")
            num_train_epochs = 1
        config["training"]["max_train_steps"] = len(train_dataloader) * num_train_epochs
        accelerator.print(f"Calculated max_train_steps: {config['training']['max_train_steps']} (from {num_train_epochs} epochs and dataloader length {len(train_dataloader)})")
    else:
        accelerator.print(f"Using pre-defined max_train_steps: {config['training']['max_train_steps']}")

    # Define resume_step
    resume_step = 0  # Assuming new training, can be loaded from checkpoint if resuming

    # Define memory_mgr (passing None as it's not actively used in the provided training_loop.py)
    memory_mgr = None

    # Define mixed_precision_dtype
    mixed_precision_dtype = None
    if accelerator.mixed_precision == "fp16":
        mixed_precision_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        mixed_precision_dtype = torch.bfloat16
    
    # Define loss_csv_path
    output_dir = config["paths"].get("output_dir", ".") # Ensure output_dir is defined
    loss_csv_path = os.path.join(output_dir, "detailed_loss_data.csv")
    os.makedirs(output_dir, exist_ok=True) # Ensure output directory exists for the csv

    # Execute the training loop
    # Pass the accelerator and the target_device for clarity if needed inside,
    # or ensure execute_training_loop uses accelerator.device
    global_step, returned_loss_history, training_successful = execute_training_loop(
        accelerator=accelerator,
        unet=unet,
        text_encoder=text_encoder,
        vae=vae,
        tokenizer=tokenizer,
        dataloader=train_dataloader,
        optimizer=optimizer,
        noise_scheduler=noise_scheduler,
        lr_scheduler=lr_scheduler,
        config=config,
        resume_step=resume_step,
        memory_mgr=memory_mgr,
        debug_monitor=debug_monitor,
        loss_monitor=loss_monitor,
        mixed_precision_dtype=mixed_precision_dtype,
        loss_csv_path=loss_csv_path,
        text_encoder_2=text_encoder_2
    )
    
    if accelerator.is_main_process:
        # Plot loss curves if history exists from LossMonitor
        loss_history_data = getattr(loss_monitor, 'loss_history', None)
        # If LossMonitor's history is empty or not preferred, could use returned_loss_history
        # if not (isinstance(loss_history_data, dict) and loss_history_data.get("steps"