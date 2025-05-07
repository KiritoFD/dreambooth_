"""
DreamBooth 训练循环模块
包含训练过程中的核心循环逻辑，负责损失计算和优化
"""
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
import os
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import csv


def execute_training_loop(
    accelerator, unet, text_encoder, vae, tokenizer, 
    dataloader, optimizer, noise_scheduler, lr_scheduler,
    config,  # Pass the entire config object
    resume_step,  # Explicitly pass the step to resume from
    memory_mgr, debug_monitor, loss_monitor,
    mixed_precision_dtype,
    loss_csv_path  # 添加loss_csv_path参数
):
    """执行DreamBooth核心训练循环"""
    
    # 检查TensorBoard日志配置
    if "report_to" in config["logging_saving"] and "tensorboard" in config["logging_saving"]["report_to"]:
        if "logging_dir" not in config["logging_saving"]:
            error_msg = (
                "配置错误: 使用TensorBoard日志需要设置logging_dir参数。\n"
                "请在config.json的logging_saving部分添加logging_dir字段，如:\n"
                '"logging_dir": "./logs"'
            )
            accelerator.print(error_msg)
            raise ValueError(error_msg)
    
    # 增强型参数检查和修复
    # 检查所有潜在的参数，寻找真正的 noise_scheduler
    potential_noise_schedulers = []
    
    # 检查函数参数中的对象是否符合 noise_scheduler 特征
    for param_name, param_val in {
        'noise_scheduler': noise_scheduler,
        'lr_scheduler': lr_scheduler,
        'optimizer': optimizer,
        'unet': unet,
        'text_encoder': text_encoder,
        'vae': vae
    }.items():
        if hasattr(param_val, 'config') and hasattr(param_val.config, 'num_train_timesteps'):
            accelerator.print(f"Found potential noise_scheduler in parameter '{param_name}'")
            potential_noise_schedulers.append((param_name, param_val))
    
    # 如果找到符合特征的对象，使用第一个作为 noise_scheduler
    if potential_noise_schedulers and not (hasattr(noise_scheduler, 'config') and hasattr(noise_scheduler.config, 'num_train_timesteps')):
        selected_name, selected_scheduler = potential_noise_schedulers[0]
        accelerator.print(f"Auto-fixing: Using '{selected_name}' as noise_scheduler")
        noise_scheduler = selected_scheduler
    
    # 最终验证
    if not hasattr(noise_scheduler, 'config') or not hasattr(noise_scheduler.config, 'num_train_timesteps'):
        # 尝试通过配置文件创建一个新的 DDPMScheduler
        try:
            from diffusers import DDPMScheduler
            pretrained_model_path = config["paths"]["pretrained_model_name_or_path"]
            accelerator.print(f"尝试从 '{pretrained_model_path}' 重新加载 DDPMScheduler")
            
            # 尝试从预训练模型创建新的 noise_scheduler
            try:
                noise_scheduler = DDPMScheduler.from_pretrained(
                    pretrained_model_path, 
                    subfolder="scheduler"
                )
                accelerator.print("成功创建了新的 DDPMScheduler")
            except Exception as load_err:
                accelerator.print(f"加载 DDPMScheduler 失败: {load_err}")
                
                # 尝试创建一个默认的 DDPMScheduler
                try:
                    accelerator.print("尝试创建默认 DDPMScheduler")
                    noise_scheduler = DDPMScheduler(
                        beta_start=0.00085, 
                        beta_end=0.012, 
                        beta_schedule="scaled_linear", 
                        num_train_timesteps=1000
                    )
                    accelerator.print("成功创建了默认 DDPMScheduler")
                except Exception as default_err:
                    accelerator.print(f"创建默认 DDPMScheduler 失败: {default_err}")
                    raise TypeError(f"无法创建有效的 noise_scheduler. 请检查 dreambooth.py 中的参数传递")
                
        except ImportError:
            accelerator.print("无法导入 DDPMScheduler，无法自动修复")
            error_msg = (
                f"CRITICAL ERROR: 找不到有效的 noise_scheduler。"
                f"当前 'noise_scheduler' 参数类型为 {type(noise_scheduler)}，"
                "它缺少 '.config.num_train_timesteps' 属性。"
                "请检查 dreambooth.py 中的调用，确保正确传递了 DDPMScheduler 对象。"
            )
            accelerator.print(error_msg)
            raise TypeError(error_msg)
    
    # 确保loss_csv_path的目录存在
    os.makedirs(os.path.dirname(os.path.abspath(loss_csv_path)), exist_ok=True)
    
    # 初始化global_step和loss_history
    global_step = resume_step
    
    # 检查现有的loss历史文件
    if os.path.exists(loss_csv_path) and global_step > 0:
        # 加载现有的loss历史
        loss_history = {"instance": [], "class": [], "total": [], "steps": []}
        with open(loss_csv_path, 'r') as f:
            import csv
            reader = csv.DictReader(f)
            for row in reader:
                step = int(row["Step"])
                if step < global_step:  # 只加载到恢复点之前的记录
                    loss_history["steps"].append(step)
                    loss_history["total"].append(float(row["Total Loss"]))
                    loss_history["instance"].append(float(row["Instance Loss"]))
                    loss_history["class"].append(float(row["Class Loss"]))
        accelerator.print(f"已加载{len(loss_history['steps'])}条历史损失记录")
    else:
        # 如果文件不存在或从头开始，初始化空的loss历史
        loss_history = {"instance": [], "class": [], "total": [], "steps": []}
        
        # 创建新的CSV文件并写入表头
        with open(loss_csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Step', 'Total Loss', 'Instance Loss', 'Class Loss'])
        accelerator.print(f"创建了新的损失记录文件: {loss_csv_path}")
    
    # Extract parameters from config
    instance_prompt = config["dataset"]["instance_prompt"]
    class_prompt = config["dataset"]["class_prompt"]
    max_train_steps = config["training"]["max_train_steps"]
    output_dir = config["paths"]["output_dir"]
    prior_preservation_weight = config["training"]["prior_loss_weight"] if config["training"]["prior_preservation"] else 0.0
    train_text_encoder = config["training"]["train_text_encoder"]
    max_grad_norm = config["training"]["max_grad_norm"]
    monitor_loss_flag = config["advanced"]["monitor_loss_in_loop"]
    save_steps = config["logging_saving"]["save_steps"]
    log_every_n_steps = config["logging_saving"]["log_every_n_steps"]
    print_status_every_n_steps = config["logging_saving"]["print_status_every_n_steps"]
    early_stop_threshold = config["training"]["early_stop_threshold"]

    if torch.cuda.is_available():
        starting_gpu_memory = torch.cuda.memory_allocated() / 1024**3
        accelerator.print(f"开始训练循环前GPU内存占用: {starting_gpu_memory:.2f} GB")
    
    if global_step > 0:
        accelerator.print(f"从步骤 {global_step} 恢复训练。")
    elif global_step >= max_train_steps:
        accelerator.print(f"警告: 从步骤 {global_step} 恢复训练，该步骤已达到或超过当前设置的最大训练步骤 {max_train_steps}.")
    
    progress_bar = tqdm(range(global_step, max_train_steps), 
                        initial=global_step, total=max_train_steps,
                        disable=not accelerator.is_local_main_process,
                        desc="训练进度")
        
    # 定义 device 变量
    device = accelerator.device
    
    # 检查设备问题并自动修复
    # 确保所有的主要组件都在同一个设备上
    accelerator.print(f"检查设备一致性. 主设备: {device}")
    components = {
        "unet": unet,
        "text_encoder": text_encoder,
        "vae": vae,
        "optimizer": optimizer
    }
    # 检查主要模型组件
    for name, component in components.items():
        if hasattr(component, "device"):
            comp_device = component.device
            if str(comp_device) != str(device):
                accelerator.print(f"警告: {name} 在 {comp_device} 上，而不是主设备 {device}. 尝试修复...")
                try:
                    if hasattr(component, "to"):
                        component.to(device)
                    accelerator.print(f"已将 {name} 移至设备 {device}")
                except Exception as e:
                    accelerator.print(f"无法将 {name} 移至设备 {device}: {e}")
    
    # 检测 UNet dtype 一次性完成，而不是在每个批次中
    unet_dtype = None
    for param in unet.parameters():
        unet_dtype = param.dtype
        break  # 只需要找到第一个参数的类型
    
    # 如果无法获取 unet 的数据类型，就使用提供的混合精度类型
    if unet_dtype is None:
        unet_dtype = mixed_precision_dtype
    
    accelerator.print(f"检测到 UNet 使用的 dtype: {unet_dtype}, 将强制所有输入与其匹配。")

    instance_text_inputs = tokenizer(
        instance_prompt, padding="max_length", 
        max_length=tokenizer.model_max_length,
        truncation=True, return_tensors="pt"
    ).to(device)
    
    # 特别检查 instance_text_inputs 的设备
    if hasattr(instance_text_inputs, "input_ids") and hasattr(instance_text_inputs.input_ids, "device"):
        input_device = instance_text_inputs.input_ids.device
        if str(input_device) != str(device):
            accelerator.print(f"警告: instance_text_inputs.input_ids 在 {input_device} 上，而不是主设备 {device}. 尝试修复...")
            try:
                instance_text_inputs.input_ids = instance_text_inputs.input_ids.to(device)
                if hasattr(instance_text_inputs, "attention_mask"):
                    instance_text_inputs.attention_mask = instance_text_inputs.attention_mask.to(device)
                accelerator.print(f"已将 instance_text_inputs 移至设备 {device}")
            except Exception as e:
                accelerator.print(f"无法将 instance_text_inputs 移至设备 {device}: {e}")
    
    class_text_inputs = tokenizer(
        class_prompt, padding="max_length",
        max_length=tokenizer.model_max_length, 
        truncation=True, return_tensors="pt"
    ).to(device) if class_prompt and prior_preservation_weight > 0 else None
    
    # 如果使用了类别提示，也检查 class_text_inputs
    if class_text_inputs is not None and hasattr(class_text_inputs, "input_ids") and hasattr(class_text_inputs.input_ids, "device"):
        input_device = class_text_inputs.input_ids.device
        if str(input_device) != str(device):
            accelerator.print(f"警告: class_text_inputs.input_ids 在 {input_device} 上，而不是主设备 {device}. 尝试修复...")
            try:
                class_text_inputs.input_ids = class_text_inputs.input_ids.to(device)
                if hasattr(class_text_inputs, "attention_mask"):
                    class_text_inputs.attention_mask = class_text_inputs.attention_mask.to(device)
                accelerator.print(f"已将 class_text_inputs 移至设备 {device}")
            except Exception as e:
                accelerator.print(f"无法将 class_text_inputs 移至设备 {device}: {e}")
    
    params_to_optimize = list(unet.parameters())
    if train_text_encoder:
        params_to_optimize += list(text_encoder.parameters())
    
    try:
        for epoch in range(1):  # 通常一个epoch就足够
            unet.train()
            if train_text_encoder:
                text_encoder.train()
            else:
                if hasattr(text_encoder, 'module') and isinstance(text_encoder.module, torch.nn.Module):
                    text_encoder.module.eval()
                elif isinstance(text_encoder, torch.nn.Module):
                    text_encoder.eval()
            
            dataset_size = len(dataloader.dataset)
            print(f"开始训练，数据集包含 {dataset_size} 个样本")
            
            for step, batch in enumerate(dataloader):
                with accelerator.accumulate(unet):  # Accumulate on unet or list of models
                    loss, instance_loss_val, class_loss_val = compute_loss(
                        accelerator, batch, unet, vae, noise_scheduler,
                        instance_text_inputs, class_text_inputs if prior_preservation_weight > 0 else None, 
                        text_encoder, prior_preservation_weight, accelerator.device, 
                        mixed_precision_dtype, unet_dtype,  # 传递 unet_dtype 而不是再次检测
                        config  # Pass config to compute_loss if it needs specific flags
                    )
                    
                    if torch.isnan(loss) or torch.isinf(loss):
                        accelerator.print("CRITICAL: Loss is NaN or Inf. Skipping step.")
                        continue
                    
                    # 每步实时打印损失值，而不是只在特定步骤打印
                    if accelerator.is_main_process:
                        # 提取损失值（如果是张量）
                        total_loss_val = loss.item() if isinstance(loss, torch.Tensor) else loss
                        inst_loss = instance_loss_val.item() if isinstance(instance_loss_val, torch.Tensor) else instance_loss_val
                        class_loss = class_loss_val.item() if isinstance(class_loss_val, torch.Tensor) else class_loss_val
                        
                        # 构建实时损失显示
                        loss_str = f"Step {global_step}/{max_train_steps} | Loss: {total_loss_val:.6f}"
                        
                        # 仅当有意义时添加实例/类别损失
                        if inst_loss != 0:
                            loss_str += f" | 实例损失: {inst_loss:.6f}"
                        if class_loss != 0:
                            loss_str += f" | 类别损失: {class_loss:.6f}"
                            
                        # 显示进度百分比
                        progress_pct = (global_step / max_train_steps) * 100
                        loss_str += f" | 进度: {progress_pct:.2f}%"
                        
                        # 打印当前批次的is_instance分布情况
                        if 'is_instance' in batch:
                            is_instance = batch['is_instance']
                            instance_count = torch.sum(is_instance).item()
                            class_count = is_instance.size(0) - instance_count
                            loss_str += f" | 批次构成: {instance_count}实例/{class_count}类别"
                        
                        accelerator.print(loss_str)
                    
                    # 修改：记录每一步的损失，不再使用log_every_n_steps的条件
                    if accelerator.is_main_process:
                        log_losses(
                            accelerator, loss, instance_loss_val, class_loss_val,
                            global_step, max_train_steps, loss_history, debug_monitor, lr_scheduler, optimizer
                        )
                    
                    if global_step % print_status_every_n_steps == 0 and global_step > 0 and accelerator.is_main_process:
                        print_training_status(
                            global_step, max_train_steps, loss, instance_loss_val, class_loss_val, prior_preservation_weight
                        )
                    
                    accelerator.backward(loss)
                    
                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(params_to_optimize, max_grad_norm)
                    
                    optimizer.step()
                    optimizer.zero_grad()
                
                # 修改：每一步都保存损失值到CSV文件
                if accelerator.is_main_process:
                    append_loss_to_csv(
                        loss_csv_path,
                        step=global_step,
                        total_loss=loss.item() if isinstance(loss, torch.Tensor) else loss,
                        instance_loss=instance_loss_val.item() if isinstance(instance_loss_val, torch.Tensor) else instance_loss_val,
                        class_loss=class_loss_val.item() if isinstance(class_loss_val, torch.Tensor) else class_loss_val
                    )
                
                # 每隔一定步数更新损失曲线
                if accelerator.is_main_process and global_step % 50 == 0 and global_step > 0:
                    update_loss_plot(loss_history, output_dir, global_step, max_train_steps)
                
                if accelerator.is_main_process:
                    if global_step % save_steps == 0 and global_step > 0:  # Use save_steps from config
                        save_checkpoint(
                            accelerator, unet, text_encoder, optimizer,
                            global_step, os.path.join(output_dir, "checkpoint.pt"), train_text_encoder
                        )
                
                progress_bar.update(1)
                global_step += 1
                
                if global_step >= max_train_steps:
                    break
        
        # 训练结束后生成最终的损失曲线
        if accelerator.is_main_process:
            update_loss_plot(loss_history, output_dir, global_step, max_train_steps)
        
        # 训练成功完成
        training_successful = True
        if accelerator.is_main_process:
            # 打印最终的训练统计信息
            if torch.cuda.is_available():
                final_gpu_memory = torch.cuda.memory_allocated() / 1024**3
                max_gpu_memory = torch.cuda.max_memory_allocated() / 1024**3
                accelerator.print(f"\n训练完成统计:\n"
                                  f"- 最终GPU内存占用: {final_gpu_memory:.2f} GB\n"
                                  f"- 训练过程中最大内存占用: {max_gpu_memory:.2f} GB\n"
                                  f"- 完成步数: {global_step}/{max_train_steps}\n"
                                  f"- 实例损失最终值: {loss_history['instance'][-1] if loss_history['instance'] else 'N/A'}\n"
                                  f"- 类别损失最终值: {loss_history['class'][-1] if loss_history['class'] else 'N/A'}")
            
            # 保存最终检查点
            save_checkpoint(
                accelerator, unet, text_encoder, optimizer,
                global_step, os.path.join(output_dir, "final_checkpoint.pt"), train_text_encoder
            )
        
    except KeyboardInterrupt:
        accelerator.print("\n训练被用户中断")
        handle_training_interruption(
            accelerator, unet, text_encoder, optimizer, 
            global_step, output_dir, train_text_encoder, "interrupt"
        )
        training_successful = False
        
    except torch.cuda.OutOfMemoryError as oom_error:
        accelerator.print(f"\n训练因GPU内存不足而中断: {str(oom_error)}")
        accelerator.print("建议: 减小批量大小、使用更小的模型或启用梯度检查点")
        handle_training_interruption(
            accelerator, unet, text_encoder, optimizer, 
            global_step, output_dir, train_text_encoder, "oom_error"
        )
        training_successful = False
        
    except Exception as e:
        accelerator.print(f"\n训练遇到错误: {str(e)}")
        import traceback
        traceback.print_exc()
        handle_training_interruption(
            accelerator, unet, text_encoder, optimizer, 
            global_step, output_dir, train_text_encoder, "error"
        )
        training_successful = False
    
    accelerator.print(f"\n训练{'成功' if training_successful else '未成功'}完成，总步数: {global_step}/{max_train_steps}")
    
    # 如果训练步骤太少，添加警告
    if global_step < max_train_steps * 0.01:  # 如果完成的步骤少于1%
        accelerator.print("\n⚠️ 警告: 训练步骤数过少，可能未能有效训练模型")
        accelerator.print("可能的原因:")
        accelerator.print("  - GPU内存不足")
        accelerator.print("  - 训练速度过慢（每步耗时过长）")
        accelerator.print("  - 数据集问题（实例图片太少或加载问题）")
        accelerator.print("  - 手动中断训练")
    
    return global_step, loss_history, training_successful


def compute_loss(
    accelerator, batch, unet, vae, noise_scheduler,
    instance_text_inputs, class_text_inputs, text_encoder, 
    prior_preservation_weight, device, 
    mixed_precision_dtype, unet_dtype,  # 添加 unet_dtype 作为参数
    config=None
):
    """计算DreamBooth训练损失"""
    try:
        # Add a check for the noise_scheduler type/attributes
        if not hasattr(noise_scheduler, 'config') or not hasattr(noise_scheduler.config, 'num_train_timesteps'):
            error_msg = (
                f"CRITICAL ERROR: The 'noise_scheduler' object (type: {type(noise_scheduler)}) "
                "does not have the expected '.config.num_train_timesteps' attribute. "
                "This usually means the wrong object (e.g., a learning rate scheduler like LambdaLR) "
                "was passed as the 'noise_scheduler' argument to 'execute_training_loop' or 'compute_loss'. "
                "Please check the call site in 'dreambooth.py' to ensure the DDPMScheduler (or similar) "
                "is passed as 'noise_scheduler' and the learning rate scheduler is passed as 'lr_scheduler'."
            )
            accelerator.print(error_msg)
            raise TypeError(error_msg)

        # Ensure mixed_precision_dtype is a torch.dtype, not a string.
        if isinstance(mixed_precision_dtype, str):
            accelerator.print(f"CRITICAL ERROR in compute_loss: mixed_precision_dtype is a string ('{mixed_precision_dtype}'), not a torch.dtype. This should have been converted earlier.")
            if mixed_precision_dtype == "fp16":
                mixed_precision_dtype = torch.float16
            elif mixed_precision_dtype == "bf16":
                mixed_precision_dtype = torch.bfloat16
            else:
                mixed_precision_dtype = torch.float32

        # 采取更加一致和严格的数据类型管理
        # 注意：设置统一的数据类型，因为模型内部可能存在混合精度的情况
        # 将 unet 和相关输入强制为相同数据类型
        pixel_values = batch["pixel_values"].to(device=device, dtype=torch.float32)  # VAE 始终使用 float32
        is_instance = batch["is_instance"].to(device)
        
        # 调试: 打印 is_instance 的具体值
        if accelerator.is_main_process:
            accelerator.print(f"[DEBUG] is_instance values: {is_instance}")

        if pixel_values.shape[0] == 0:
            return torch.tensor(0.0, device=device, dtype=mixed_precision_dtype, requires_grad=False), 0.0, 0.0
        
        # VAE 总是使用 float32
        pixel_values_for_vae = pixel_values.to(dtype=torch.float32)
        with torch.no_grad():
            latents = vae.encode(pixel_values_for_vae).latent_dist.sample()
            latents = latents * vae.config.scaling_factor
        
        # 现在与 unet 的数据类型匹配
        latents = latents.to(dtype=unet_dtype)

        noise = torch.randn_like(latents)  # 这将继承 latents 的数据类型
        
        # 确保时间步数据也匹配 UNet 参数的数据类型
        timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, 
                                 (latents.shape[0],), device=latents.device).long()
        
        # 确保所有时间步都是长整型
        timesteps = timesteps.long()
        
        noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
        
        instance_emb = None
        class_emb = None
        
        if torch.any(is_instance):
            with torch.set_grad_enabled(config["training"]["train_text_encoder"]):
                instance_emb_raw = text_encoder(instance_text_inputs.input_ids)[0]
            # 确保它与 unet 数据类型匹配
            instance_emb = instance_emb_raw.to(dtype=unet_dtype)

        if class_text_inputs is not None and torch.any(~is_instance) and prior_preservation_weight > 0:
            with torch.set_grad_enabled(config["training"]["train_text_encoder"]):
                class_emb_raw = text_encoder(class_text_inputs.input_ids)[0]
            # 确保它与 unet 数据类型匹配
            class_emb = class_emb_raw.to(dtype=unet_dtype)

        instance_loss = 0.0
        class_loss = 0.0
        
        instance_count = torch.sum(is_instance).item()
        class_count = torch.sum(~is_instance).item()

        if instance_count > 0:
            # 确保 timesteps 的类型是 long，而不是 UNet 期望的其他类型
            instance_pred = unet(
                noisy_latents[is_instance],
                timesteps[is_instance],
                encoder_hidden_states=instance_emb.repeat(instance_count, 1, 1)
            ).sample
            
            # MSE 损失计算应该使用 float32 以提高数值稳定性
            instance_loss_tensor = F.mse_loss(
                instance_pred.float(),  # 转换为 float32 用于计算损失
                noise[is_instance].float(),
                reduction="mean"
            )
            instance_loss = instance_loss_tensor
        
        if class_count > 0:
            # 确保所有输入到 unet 的数据都是正确的数据类型
            class_pred = unet(
                noisy_latents[~is_instance].to(dtype=unet_dtype),  # 显式转换为与 unet 相同的数据类型
                timesteps[~is_instance].long(),  # 确保时间步是长整数
                encoder_hidden_states=class_emb.repeat(class_count, 1, 1).to(dtype=unet_dtype)  # 显式转换数据类型
            ).sample
            
            # MSE 损失计算应该使用 float32 以提高数值稳定性
            class_loss = F.mse_loss(
                class_pred.float(),  # 转换为 float32 用于计算损失
                noise[~is_instance].float(),
                reduction="mean"
            )
        
        if isinstance(instance_loss, torch.Tensor) and isinstance(class_loss, torch.Tensor):
            loss = instance_loss + prior_preservation_weight * class_loss
        elif isinstance(instance_loss, torch.Tensor):
            loss = instance_loss
        elif isinstance(class_loss, torch.Tensor):
            loss = prior_preservation_weight * class_loss
        else:
            loss = torch.tensor(0.0, device=device, dtype=mixed_precision_dtype)
        
        return loss, instance_loss, class_loss
    
    except Exception as e:
        accelerator.print(f"损失计算错误: {e}")
        import traceback
        accelerator.print(traceback.format_exc())
        error_loss = torch.tensor(float('inf'), device=device, dtype=torch.float32)
        return error_loss, 0.0, 0.0


def log_losses(accelerator, loss, instance_loss, class_loss, global_step, max_train_steps, loss_history, debug_monitor=None, lr_scheduler=None, optimizer=None):
    """记录训练损失"""
    il = instance_loss.item() if isinstance(instance_loss, torch.Tensor) else instance_loss
    cl = class_loss.item() if isinstance(class_loss, torch.Tensor) else class_loss
    tl = loss.detach().item()
    
    loss_history["instance"].append(float(il))
    loss_history["class"].append(float(cl))
    loss_history["total"].append(float(tl))
    loss_history["steps"].append(global_step)
    
    # 添加训练进度打印
    progress = global_step / max_train_steps * 100
    
    # 获取当前学习率
    current_lr = "N/A"
    if lr_scheduler:
        try:
            current_lr = lr_scheduler.get_last_lr()[0]
        except AttributeError:
            try:
                current_lr = optimizer.param_groups[0]['lr']
            except:
                pass
    
    # 继续原有的监控逻辑
    if debug_monitor and accelerator.is_main_process:
        debug_monitor.log_step(global_step, max_train_steps, {
            "instance_loss": il,
            "class_loss": cl,
            "total_loss": tl,
            "lr": f"{current_lr:.2e}" if isinstance(current_lr, float) else current_lr 
        })


def print_training_status(global_step, max_train_steps, loss, instance_loss, class_loss, prior_preservation_weight):
    """打印详细训练状态"""
    # 简化为只打印进度信息，因为损失已经实时打印
    print(f"\n训练进度: {global_step/max_train_steps*100:.1f}% ({global_step}/{max_train_steps})")


def save_checkpoint(accelerator, unet, text_encoder, optimizer, global_step, checkpoint_path, train_text_encoder=True):
    """保存训练检查点"""
    checkpoint = {
        "global_step": global_step,
        "optimizer": optimizer.state_dict(),
        "unet": accelerator.unwrap_model(unet).state_dict(),
    }
    if train_text_encoder:
        checkpoint["text_encoder"] = accelerator.unwrap_model(text_encoder).state_dict()
    
    torch.save(checkpoint, checkpoint_path)
    accelerator.print(f"保存检查点到步骤 {global_step}")


def handle_training_interruption(accelerator, unet, text_encoder, optimizer, global_step, output_dir, train_text_encoder, reason="interrupt"):
    """处理训练中断情况"""
    accelerator.print("\n尝试保存当前模型状态...")
    
    if accelerator.is_local_main_process:
        try:
            checkpoint = {
                "global_step": global_step,
                "optimizer": optimizer.state_dict(),
                "unet": accelerator.unwrap_model(unet).state_dict(),
            }
            if train_text_encoder:
                checkpoint["text_encoder"] = accelerator.unwrap_model(text_encoder).state_dict()
            
            checkpoint_path = os.path.join(output_dir, f"{reason}_checkpoint.pt")
            torch.save(checkpoint, checkpoint_path)
            accelerator.print(f"{reason.capitalize()}检查点已保存，您可以稍后恢复训练")
            return True
        except:
            accelerator.print(f"保存{reason}检查点失败")
    
    return False


# 添加动态绘制损失曲线的函数
def update_loss_plot(loss_history, output_dir, global_step, max_train_steps):
    """实时更新损失曲线并保存"""
    try:
        import matplotlib.pyplot as plt
        from matplotlib.ticker import MaxNLocator
        
        # 如果数据点少于2，无法绘图
        if len(loss_history["steps"]) < 2:
            return
            
        # 创建图表
        plt.figure(figsize=(10, 6))
        
        # 绘制各类损失曲线
        plt.plot(loss_history["steps"], loss_history["total"], 'b-', label='总损失')
        if any(x != 0 for x in loss_history["instance"]):
            plt.plot(loss_history["steps"], loss_history["instance"], 'g-', label='实例损失')
        if any(x != 0 for x in loss_history["class"]):
            plt.plot(loss_history["steps"], loss_history["class"], 'r-', label='类别损失')
        
        # 添加标题和标签
        plt.title(f'DreamBooth训练损失 (进度: {global_step}/{max_train_steps})')
        plt.xlabel('训练步骤')
        plt.ylabel('损失值')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # 确保X轴使用整数刻度
        plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
        
        # 保存图表
        save_path = os.path.join(output_dir, "live_loss_curve.png")
        plt.savefig(save_path, dpi=100)
        plt.close()
        
    except Exception as e:
        print(f"绘制损失曲线时出错: {str(e)}")


def append_loss_to_csv(csv_path, step, total_loss, instance_loss, class_loss):
    """将损失值追加到CSV文件"""
    try:
        # 确保目录存在
        os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)
        
        file_exists = os.path.isfile(csv_path)
        with open(csv_path, mode='a', newline='') as file:
            writer = csv.writer(file)
            if not file_exists:
                writer.writerow(["Step", "Total Loss", "Instance Loss", "Class Loss"])
            writer.writerow([step, total_loss, instance_loss, class_loss])
            
        # 确保数据立即写入磁盘
        file.flush()
        os.fsync(file.fileno())
    except Exception as e:
        print(f"写入损失值到CSV时出错: {e}")
