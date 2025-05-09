import os
import csv
import torch
from tqdm.auto import tqdm
from diffusers import DDPMScheduler # For default scheduler creation

def initialize_training_environment(
    accelerator, unet, text_encoder, vae, tokenizer, optimizer, 
    noise_scheduler_in, # Renamed to avoid conflict
    config, resume_step, mixed_precision_dtype, loss_csv_path, dataloader
):
    """Initializes the training environment, components, and parameters."""

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
    # Use noise_scheduler_in for the check
    for param_name, param_val in {
        'noise_scheduler': noise_scheduler_in,
        # 'lr_scheduler': lr_scheduler, # lr_scheduler is not passed to this function yet
        'optimizer': optimizer,
        'unet': unet,
        'text_encoder': text_encoder,
        'vae': vae
    }.items():
        if hasattr(param_val, 'config') and hasattr(param_val.config, 'num_train_timesteps'):
            accelerator.print(f"Found potential noise_scheduler in parameter '{param_name}'")
            potential_noise_schedulers.append((param_name, param_val))
    
    current_noise_scheduler = noise_scheduler_in
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
            # ... (error message)
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
        "log_every_n_steps": config["logging_saving"]["log_every_n_steps"], # Not directly used in the loop after refactor, but kept for completeness
        "print_status_every_n_steps": config["logging_saving"]["print_status_every_n_steps"],
        "early_stop_threshold": config["training"]["early_stop_threshold"], # Not directly used in the loop after refactor
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
    
    device = accelerator.device
    accelerator.print(f"检查设备一致性. 主设备: {device}")
    components_to_check = {"unet": unet, "text_encoder": text_encoder, "vae": vae, "optimizer": optimizer}
    for name, component in components_to_check.items():
        if hasattr(component, "device"): # Optimizer might not have .device directly, its params do.
            # For optimizer, check params' devices if component.device is not present
            comp_device_str = ""
            if hasattr(component, "param_groups"): # It's an optimizer
                if component.param_groups and component.param_groups[0]['params']:
                     comp_device_str = str(component.param_groups[0]['params'][0].device)
            elif hasattr(component, "device"):
                 comp_device_str = str(component.device)

            if comp_device_str and comp_device_str != str(device):
                #accelerator.print(f"警告: {name} 参数可能在 {comp_device_str} 上，而不是主设备 {device}. 尝试修复...")
                try:
                    if hasattr(component, "to"): # Models have .to()
                        component.to(device)
                        #accelerator.print(f"已将 {name} 移至设备 {device}")
                except Exception as e:
                    accelerator.print(f"无法将 {name} 移至设备 {device}: {e}")
    
    unet_dtype_val = None
    for param in unet.parameters():
        unet_dtype_val = param.dtype
        break
    if unet_dtype_val is None:
        unet_dtype_val = mixed_precision_dtype
    accelerator.print(f"检测到 UNet 使用的 dtype: {unet_dtype_val}, 将强制所有输入与其匹配。")

    instance_text_inputs = tokenizer(
        params["instance_prompt"], padding="max_length", 
        max_length=tokenizer.model_max_length,
        truncation=True, return_tensors="pt"
    ).to(device)
    # Device check for tokenized inputs
    if hasattr(instance_text_inputs, "input_ids") and hasattr(instance_text_inputs.input_ids, "device"):
        if str(instance_text_inputs.input_ids.device) != str(device):
            accelerator.print(f"警告: instance_text_inputs.input_ids 在 {instance_text_inputs.input_ids.device} 上... 尝试修复.")
            instance_text_inputs = instance_text_inputs.to(device)


    class_text_inputs = None
    if params["class_prompt"] and params["prior_preservation_weight"] > 0:
        class_text_inputs = tokenizer(
            params["class_prompt"], padding="max_length",
            max_length=tokenizer.model_max_length, 
            truncation=True, return_tensors="pt"
        ).to(device)
        if hasattr(class_text_inputs, "input_ids") and hasattr(class_text_inputs.input_ids, "device"):
             if str(class_text_inputs.input_ids.device) != str(device):
                accelerator.print(f"警告: class_text_inputs.input_ids 在 {class_text_inputs.input_ids.device} 上... 尝试修复.")
                class_text_inputs = class_text_inputs.to(device)


    return {
        "noise_scheduler": current_noise_scheduler,
        "global_step": global_step,
        "loss_history": loss_history,
        "config_params": params,
        "progress_bar": progress_bar,
        "image_progress": image_progress,
        "device": device,
        "unet_dtype": unet_dtype_val,
        "instance_text_inputs": instance_text_inputs,
        "class_text_inputs": class_text_inputs,
    }
