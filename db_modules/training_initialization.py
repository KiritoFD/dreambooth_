import os
import csv
import torch
import gc
from tqdm.auto import tqdm
from diffusers import DDPMScheduler # For default scheduler creation

def initialize_training_environment(
    accelerator, unet, text_encoder, vae, tokenizer, optimizer, noise_scheduler, 
    config, resume_step, mixed_precision_dtype, loss_csv_path, dataloader,
    text_encoder_2=None  # 添加第二个文本编码器参数
):
    """初始化训练环境，准备和配置所需的所有变量，如提示器和进度条"""
    
    #检查TensorBoard日志配置
    if "report_to" in config["logging_saving"] and "tensorboard" in config["logging_saving"]["report_to"]:
        if "logging_dir" not in config["logging_saving"]:
            error_msg = (
                "配置错误: 使用TensorBoard日志需要设置logging_dir参数。\n"
                "请在config.json的logging_saving部分添加logging_dir字段，如:\n"
                '"logging_dir": "./logs"'
            )
            accelerator.print(error_msg)
            raise ValueError(error_msg)

    # 增强型参数检查和修复 for noise_scheduler
    potential_noise_schedulers = []
    for param_name, param_val in {
        'noise_scheduler': noise_scheduler,
        'optimizer': optimizer,
        'unet': unet,
        'text_encoder': text_encoder,
        'vae': vae
    }.items():
        if hasattr(param_val, 'config') and hasattr(param_val.config, 'num_train_timesteps'):
            accelerator.print(f"Found potential noise_scheduler in parameter '{param_name}'")
            potential_noise_schedulers.append((param_name, param_val))
    
    current_noise_scheduler = noise_scheduler
    if potential_noise_schedulers and not (hasattr(current_noise_scheduler, 'config') and hasattr(current_noise_scheduler.config, 'num_train_timesteps')):
        selected_name, selected_scheduler = potential_noise_schedulers[0]
        accelerator.print(f"Auto-fixing: Using '{selected_name}' as noise_scheduler")
        current_noise_scheduler = selected_scheduler
    
    if not hasattr(current_noise_scheduler, 'config') or not hasattr(current_noise_scheduler.config, 'num_train_timesteps'):
        try:
            pretrained_model_path = config["paths"]["pretrained_model_name_or_path"]
            accelerator.print(f"尝试从 '{pretrained_model_path}' 重新加载 DDPMScheduler")
            try:
                current_noise_scheduler = DDPMScheduler.from_pretrained(
                    pretrained_model_path, subfolder="scheduler"
                )
                accelerator.print("成功创建了新的 DDPMScheduler")
            except Exception as load_err:
                accelerator.print(f"加载 DDPMScheduler 失败: {load_err}")
                try:
                    accelerator.print("尝试创建默认 DDPMScheduler")
                    current_noise_scheduler = DDPMScheduler(
                        beta_start=0.00085, beta_end=0.012, 
                        beta_schedule="scaled_linear", num_train_timesteps=1000
                    )
                    accelerator.print("成功创建了默认 DDPMScheduler")
                except Exception as default_err:
                    accelerator.print(f"创建默认 DDPMScheduler 失败: {default_err}")
                    raise TypeError(f"无法创建有效的 noise_scheduler. 请检查 dreambooth.py 中的参数传递")
        except ImportError:
            accelerator.print("无法导入 DDPMScheduler，无法自动修复")
            raise TypeError("Error related to DDPMScheduler import or instantiation.")

    # 确保loss_csv_path的目录存在
    os.makedirs(os.path.dirname(os.path.abspath(loss_csv_path)), exist_ok=True)
    
    global_step = resume_step
    loss_history = {"instance": [], "class": [], "total": [], "steps": []}

    if os.path.exists(loss_csv_path) and global_step > 0:
        with open(loss_csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                step_val = int(row["Step"])
                if step_val < global_step:
                    loss_history["steps"].append(step_val)
                    loss_history["total"].append(float(row["Total Loss"]))
                    loss_history["instance"].append(float(row["Instance Loss"]))
                    loss_history["class"].append(float(row["Class Loss"]))
        accelerator.print(f"已加载{len(loss_history['steps'])}条历史损失记录")
    else:
        with open(loss_csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Step', 'Total Loss', 'Instance Loss', 'Class Loss'])
        accelerator.print(f"创建了新的损失记录文件: {loss_csv_path}")

    # Extract parameters from config
    params = {
        "instance_prompt": config["dataset"]["instance_prompt"],
        "class_prompt": config["dataset"]["class_prompt"],
        "max_train_steps": config["training"]["max_train_steps"],
        "output_dir": config["paths"]["output_dir"],
        "prior_preservation_weight": config["training"]["prior_loss_weight"] if config["training"]["prior_preservation"] else 0.0,
        "train_text_encoder": config["training"]["train_text_encoder"],
        "max_grad_norm": config["training"]["max_grad_norm"],
        "save_steps": config["logging_saving"]["save_steps"],
        "log_every_n_steps": config["logging_saving"]["log_every_n_steps"],
        "print_status_every_n_steps": config["logging_saving"]["print_status_every_n_steps"],
        "early_stop_threshold": config["training"]["early_stop_threshold"],
        "batch_size": config["dataset"]["train_batch_size"]
    }

    if torch.cuda.is_available():
        starting_gpu_memory = torch.cuda.memory_allocated() / 1024**3
        accelerator.print(f"开始训练循环前GPU内存占用: {starting_gpu_memory:.2f} GB")
    
    if global_step > 0:
        accelerator.print(f"从步骤 {global_step} 恢复训练。")
    elif global_step >= params["max_train_steps"]:
        accelerator.print(f"警告: 从步骤 {global_step} 恢复训练，该步骤已达到或超过当前设置的最大训练步骤 {params['max_train_steps']}.")

    processed_images = resume_step * params["batch_size"]
    total_images_for_pbar = params["max_train_steps"] * params["batch_size"]

    progress_bar = tqdm(range(global_step, params["max_train_steps"]), 
                        initial=global_step, total=params["max_train_steps"],
                        disable=not accelerator.is_local_main_process,
                        desc="训练进度")
    image_progress = tqdm(
        initial=processed_images, total=total_images_for_pbar,
        disable=not accelerator.is_local_main_process, desc="图片处理进度"
    )
    
    # 确保所有模型都在同一个设备上
    device = accelerator.device
    
    # 创建输入提示
    if "dataset" in config and "instance_prompt" in config["dataset"]:
        instance_prompt = config["dataset"]["instance_prompt"]
    else:
        instance_prompt = "a photo of sks instance"
        accelerator.print("[WARNING] 未找到实例提示，使用默认值")
    
    # 确保分词器输出位于正确的设备上
    instance_text_inputs = tokenizer(
        instance_prompt,
        padding="do_not_pad",
        truncation=True,
        max_length=tokenizer.model_max_length,
        return_tensors="pt"
    ).to(device)
    
    class_text_inputs = None
    if config["training"].get("prior_preservation", False) and "class_prompt" in config["dataset"]:
        class_text_inputs = tokenizer(
            config["dataset"]["class_prompt"],
            padding="do_not_pad",
            truncation=True,
            max_length=tokenizer.model_max_length,
            return_tensors="pt"
        ).to(device)  # 确保在正确的设备上
    
    # 确保模型在正确的设备上 - 明确移动到设备
    if hasattr(text_encoder, 'to') and not hasattr(text_encoder, 'device'):
        text_encoder = text_encoder.to(device)
    
    if text_encoder_2 is not None and hasattr(text_encoder_2, 'to') and not hasattr(text_encoder_2, 'device'):
        text_encoder_2 = text_encoder_2.to(device)
        
    # 添加调试信息以确认设备
    if accelerator.is_main_process:
        accelerator.print(f"[DEBUG] 设备确认:")
        accelerator.print(f"- 主设备: {device}")
        accelerator.print(f"- text_encoder 设备: {next(text_encoder.parameters()).device}")
        if text_encoder_2 is not None:
            accelerator.print(f"- text_encoder_2 设备: {next(text_encoder_2.parameters()).device}")
        accelerator.print(f"- instance_text_inputs 设备: {instance_text_inputs.input_ids.device}")
        if class_text_inputs is not None:
            accelerator.print(f"- class_text_inputs 设备: {class_text_inputs.input_ids.device}")
    
    # 添加记录第二个文本编码器的状态
    if text_encoder_2 is not None:
        is_sdxl = True
        accelerator.print("第二个文本编码器已提供，配置为SDXL模型")
        
        # 如果未在advanced设置中指定model_type
        if config and "advanced" in config and "model_type" not in config["advanced"]:
            accelerator.print("自动将配置中的model_type设置为sdxl")
            config["advanced"]["model_type"] = "sdxl"
        
        # 检查编码器是否在训练模式
        train_text_encoder = config["training"]["train_text_encoder"]
        text_encoder_2.requires_grad_(train_text_encoder)
        if train_text_encoder:
            accelerator.print("第二个文本编码器设置为训练模式")
            
            # 如果启用了梯度检查点，也应用到第二个编码器
            if config["training"]["gradient_checkpointing"]:
                if hasattr(text_encoder_2, "gradient_checkpointing_enable"):
                    text_encoder_2.gradient_checkpointing_enable()
                    accelerator.print("已为第二个文本编码器启用梯度检查点")
    
    # 添加显存优化
    if config["memory_optimization"].get("aggressive_gc", False):
        # 执行初始内存清理
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            accelerator.print("已执行初始内存清理")
    
    # 调整内存使用限制以减少OOM风险
    if torch.cuda.is_available() and config["memory_optimization"].get("limit_gpu_memory", False):
        fraction = config["memory_optimization"].get("gpu_memory_fraction", 0.9)
        torch.cuda.set_per_process_memory_fraction(fraction)
        accelerator.print(f"已将GPU内存使用限制为总内存的{fraction*100}%")
    
    # 配置增强后的记录
    if accelerator.is_main_process:
        if torch.cuda.is_available():
            current_gpu_memory = torch.cuda.memory_allocated() / 1024**3
            max_gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            accelerator.print(f"当前GPU内存使用: {current_gpu_memory:.2f}GB / {max_gpu_memory:.2f}GB ({current_gpu_memory/max_gpu_memory*100:.1f}%)")

    return {
        "noise_scheduler": current_noise_scheduler,
        "global_step": global_step,
        "loss_history": loss_history,
        "config_params": params,
        "progress_bar": progress_bar,
        "image_progress": image_progress,
        "device": device,
        "unet_dtype": mixed_precision_dtype,
        "instance_text_inputs": instance_text_inputs,
        "class_text_inputs": class_text_inputs,
        "text_encoder_2": text_encoder_2  # 添加第二个文本编码器
    }
