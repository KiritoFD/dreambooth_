import os
import random
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm
from PIL import Image

from diffusers import StableDiffusionPipeline, DDPMScheduler, UNet2DConditionModel, AutoencoderKL
from diffusers.optimization import get_scheduler  # <--- 添加此行
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
        # Debugging __getitem__ - Initial call and lengths
        print(f"[DreamBoothDataset DEBUG] __getitem__ called with idx: {idx}")
        print(f"[DreamBoothDataset DEBUG] len(self.instance_images): {len(self.instance_images)}")
        print(f"[DreamBoothDataset DEBUG] len(self.class_images): {len(self.class_images)}")
        
        # 交替返回实例和类别图片的策略，确保实例图片得到足够训练
        if len(self.instance_images) > 0 and (
            self.last_was_instance == False or    # 如果上次返回的是类别图片
            idx % 5 == 0 or                      # 或者按照一定间隔强制返回实例图片
            random.random() < self.min_instance_ratio  # 或者随机概率返回实例图片
        ):
            image = self.instance_images[idx % len(self.instance_images)]
            is_instance_current = True
            self.last_was_instance = True
            print(f"[DreamBoothDataset DEBUG] 强制返回实例图片，idx: {idx}, 使用实例索引: {idx % len(self.instance_images)}")
        elif idx < len(self.instance_images):
            image = self.instance_images[idx]
            is_instance_current = True
            self.last_was_instance = True
            print(f"[DreamBoothDataset DEBUG] 正常索引返回实例图片。")
        elif idx < len(self.instance_images) + len(self.class_images):
            image = self.class_images[idx - len(self.instance_images)]
            is_instance_current = False
            self.last_was_instance = False
            print(f"[DreamBoothDataset DEBUG] 正常索引返回类别图片。")
        else:
            # This case should ideally not be hit if __len__ is correct and DataLoader respects it.
            print(f"[DreamBoothDataset DEBUG CRITICAL] Index {idx} is out of bounds. Dataset length: {len(self)}")
            raise IndexError(f"Index {idx} out of bounds for dataset with length {len(self)}")

        print(f"[DreamBoothDataset DEBUG] Determined is_instance for idx {idx}: {is_instance_current}")

        image_np = np.array(image).astype(np.float32)
        image_tensor = torch.from_numpy(image_np / 127.5 - 1.0)
        image_tensor = image_tensor.permute(2, 0, 1)
        
        return {"pixel_values": image_tensor, "is_instance": torch.tensor(is_instance_current, dtype=torch.bool)}

def load_config(config_path):
    import json
    with open(config_path, "r") as f:
        return json.load(f)

def dreambooth_training(config):
    """DreamBooth核心训练逻辑"""
    accelerator = Accelerator(
        gradient_accumulation_steps=config["training"]["gradient_accumulation_steps"],
        mixed_precision=config["training"]["mixed_precision"],
        log_with=config["logging_saving"]["report_to"],
    )

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

    # Freeze VAE
    vae.requires_grad_(False)
    # Move VAE to device, typically in float32 for stability
    vae.to(accelerator.device, dtype=torch.float32)

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
                weight_decay=config["training"]["adam_weight_decay"],
                eps=config["training"]["adam_epsilon"],
            )
        except Exception as e: # Catch any other exception during 8bit Adam creation
            accelerator.print(f"Error creating 8-bit AdamW optimizer: {e}. Using standard AdamW optimizer.")
            optimizer = torch.optim.AdamW(
                params_to_optimize,
                lr=config["training"]["learning_rate"],
                betas=(config["training"]["adam_beta1"], config["training"]["adam_beta2"]),
                weight_decay=config["training"]["adam_beta_weight_decay"],
                eps=config["training"]["adam_epsilon"],
            )
    else:
        optimizer = torch.optim.AdamW(
            params_to_optimize,
            lr=config["training"]["learning_rate"],
            betas=(config["training"]["adam_beta1"], config["training"]["adam_beta2"]),
            weight_decay=config["training"]["adam_beta_weight_decay"],
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

    # Determine the mixed_precision_dtype that accelerator is using for its models
    actual_mixed_precision_dtype = torch.float32 # Default if "no" or unrecognized
    if config["training"]["mixed_precision"] == "fp16":
        actual_mixed_precision_dtype = torch.float16
    elif config["training"]["mixed_precision"] == "bf16":
        actual_mixed_precision_dtype = torch.bfloat16
    
    accelerator.print(f"Using mixed precision: {config['training']['mixed_precision']}, corresponding to torch.dtype: {actual_mixed_precision_dtype}")

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
        resume_step=0,
        memory_mgr=None,
        debug_monitor=None,
        loss_monitor=None,
        mixed_precision_dtype=actual_mixed_precision_dtype # Pass the torch.dtype object
    )

    if accelerator.is_main_process:
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