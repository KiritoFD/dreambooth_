"""
DreamBooth 训练循环模块
包含训练过程中的核心循环逻辑，负责损失计算和优化
"""
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
import os


def execute_training_loop(
    accelerator, unet, text_encoder, vae, tokenizer, 
    dataloader, optimizer, noise_scheduler, lr_scheduler,
    config,  # Pass the entire config object
    resume_step,  # Explicitly pass the step to resume from
    memory_mgr, debug_monitor, loss_monitor,
    mixed_precision_dtype
):
    """执行DreamBooth核心训练循环"""
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
    
    global_step = resume_step  # Initialize global_step with resume_step
        
    if global_step >= max_train_steps:
        accelerator.print(f"警告: 从步骤 {global_step} 恢复训练，该步骤已达到或超过当前设置的最大训练步骤 {max_train_steps}.")
    
    progress_bar = tqdm(range(global_step, max_train_steps), 
                        initial=global_step, total=max_train_steps,
                        disable=not accelerator.is_local_main_process,
                        desc="训练进度")
        
    loss_history = {"instance": [], "class": [], "total": [], "steps": []}
    
    device = accelerator.device
    instance_text_inputs = tokenizer(
        instance_prompt, padding="max_length", 
        max_length=tokenizer.model_max_length,
        truncation=True, return_tensors="pt"
    ).to(device)
    
    class_text_inputs = tokenizer(
        class_prompt, padding="max_length",
        max_length=tokenizer.model_max_length, 
        truncation=True, return_tensors="pt"
    ).to(device) if class_prompt and prior_preservation_weight > 0 else None
    
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
                        mixed_precision_dtype,  # Dtype for UNet and trained TextEncoder
                        config  # Pass config to compute_loss if it needs specific flags
                    )
                    
                    if torch.isnan(loss) or torch.isinf(loss):
                        accelerator.print("CRITICAL: Loss is NaN or Inf. Skipping step.")
                        continue
                    
                    if accelerator.is_main_process and global_step % 10 == 0:
                        accelerator.print(f"Step {global_step}: Loss val={loss.item() if isinstance(loss, torch.Tensor) else loss}, grad_fn={loss.grad_fn if isinstance(loss, torch.Tensor) else 'N/A'}, req_grad={loss.requires_grad if isinstance(loss, torch.Tensor) else 'N/A'}")
                        if isinstance(loss, torch.Tensor) and loss.grad_fn is None and loss.requires_grad:
                            accelerator.print("Warning: Loss requires_grad but has no grad_fn. This is unusual.")
                        elif isinstance(loss, torch.Tensor) and not loss.requires_grad:
                            accelerator.print("CRITICAL: Loss does not require_grad. Gradients will not be computed.")
                    
                    if monitor_loss_flag and loss_monitor:  # Use the flag from config
                        is_loss_ok = loss_monitor.check_loss(
                            instance_loss=instance_loss_val.item() if isinstance(instance_loss_val, torch.Tensor) else instance_loss_val,
                            prior_loss=class_loss_val.item() if isinstance(class_loss_val, torch.Tensor) else class_loss_val,
                            total_loss=loss.item() if isinstance(loss, torch.Tensor) else loss,
                            step=global_step
                        )
                        if not is_loss_ok:
                            suggestions = loss_monitor.suggest_fixes()
                            accelerator.print("\n⚠️ 损失异常，建议:")
                            for suggestion in suggestions:
                                accelerator.print(f"  - {suggestion}")
                    
                    if accelerator.is_main_process and global_step % log_every_n_steps == 0:  # Use from config
                        log_losses(
                            accelerator, loss, instance_loss_val, class_loss_val,
                            global_step, max_train_steps, loss_history, debug_monitor, lr_scheduler  # Pass lr_scheduler for logging LR
                        )
                    
                    if global_step % print_status_every_n_steps == 0 and global_step > 0 and accelerator.is_main_process:  # Use from config
                        print_training_status(
                            global_step, max_train_steps, loss, instance_loss_val, class_loss_val, prior_preservation_weight
                        )
                    
                    accelerator.backward(loss)
                    
                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(params_to_optimize, max_grad_norm)
                    
                    optimizer.step()
                    optimizer.zero_grad()
                
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
        
        training_successful = True
        
    except KeyboardInterrupt:
        accelerator.print("\n训练被用户中断")
        handle_training_interruption(
            accelerator, unet, text_encoder, optimizer, 
            global_step, output_dir, train_text_encoder, "interrupt"
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
    
    return global_step, loss_history, training_successful


def compute_loss(
    accelerator, batch, unet, vae, noise_scheduler,
    instance_text_inputs, class_text_inputs, text_encoder, 
    prior_preservation_weight, device, weight_dtype=torch.float32,
    config=None  # Added config for potential future use or specific flags
):
    """计算DreamBooth训练损失"""
    try:
        pixel_values = batch["pixel_values"].to(device=device, dtype=weight_dtype)
        is_instance = batch["is_instance"].to(device)
        
        if pixel_values.shape[0] == 0:
            return torch.tensor(0.0, device=device, dtype=weight_dtype, requires_grad=False), 0.0, 0.0
        
        pixel_values_for_vae = pixel_values.to(dtype=torch.float32)
        with torch.no_grad():
            latents = vae.encode(pixel_values_for_vae).latent_dist.sample()
            latents = latents * vae.config.scaling_factor
        
        latents = latents.to(dtype=weight_dtype)

        noise = torch.randn_like(latents)
        timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, 
                                  (latents.shape[0],), device=latents.device).long()
        noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
        
        instance_emb = None
        class_emb = None
        
        if torch.any(is_instance):
            with torch.set_grad_enabled(config["training"]["train_text_encoder"]):
                instance_emb_raw = text_encoder(instance_text_inputs.input_ids)[0]
            instance_emb = instance_emb_raw.to(dtype=weight_dtype)

        if class_text_inputs is not None and torch.any(~is_instance) and prior_preservation_weight > 0:
            with torch.set_grad_enabled(config["training"]["train_text_encoder"]):
                class_emb_raw = text_encoder(class_text_inputs.input_ids)[0]
            class_emb = class_emb_raw.to(dtype=weight_dtype)

        instance_loss = 0.0
        class_loss = 0.0
        
        instance_count = torch.sum(is_instance).item()
        class_count = torch.sum(~is_instance).item()

        if instance_count > 0:
            instance_pred = unet(
                noisy_latents[is_instance],
                timesteps[is_instance],
                encoder_hidden_states=instance_emb.repeat(instance_count, 1, 1)
            ).sample
            
            instance_loss_tensor = F.mse_loss(
                instance_pred.float(),
                noise[is_instance].float(),
                reduction="mean"
            )
            instance_loss = instance_loss_tensor
        
        if class_count > 0:
            class_pred = unet(
                noisy_latents[~is_instance],
                timesteps[~is_instance],
                encoder_hidden_states=class_emb.repeat(class_count, 1, 1)
            ).sample
            
            class_loss = F.mse_loss(
                class_pred.float(),
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
            loss = torch.tensor(0.0, device=device, dtype=weight_dtype)
        
        return loss, instance_loss, class_loss
    
    except Exception as e:
        accelerator.print(f"损失计算错误: {e}")
        import traceback
        traceback.print_exc()
        zero_tensor = torch.tensor(0.0, device=device)
        return zero_tensor, 0.0, 0.0


def log_losses(accelerator, loss, instance_loss, class_loss, global_step, max_train_steps, loss_history, debug_monitor=None, lr_scheduler=None):
    """记录训练损失"""
    il = instance_loss.item() if isinstance(instance_loss, torch.Tensor) else instance_loss
    cl = class_loss.item() if isinstance(class_loss, torch.Tensor) else class_loss
    tl = loss.detach().item()
    
    loss_history["instance"].append(float(il))
    loss_history["class"].append(float(cl))
    loss_history["total"].append(float(tl))
    loss_history["steps"].append(global_step)
    
    current_lr = "N/A"
    if lr_scheduler:
        try:
            current_lr = lr_scheduler.get_last_lr()[0]
        except AttributeError:
            try:
                current_lr = optimizer.param_groups[0]['lr']
            except:
                pass

    if debug_monitor and accelerator.is_main_process: 
        debug_monitor.log_step(global_step, max_train_steps, {
            "instance_loss": il,
            "class_loss": cl,
            "total_loss": tl,
            "lr": f"{current_lr:.2e}" if isinstance(current_lr, float) else current_lr 
        })


def print_training_status(global_step, max_train_steps, loss, instance_loss, class_loss, prior_preservation_weight):
    """打印详细训练状态"""
    il_val = instance_loss.item() if isinstance(instance_loss, torch.Tensor) else instance_loss
    cl_val = class_loss.item() if isinstance(class_loss, torch.Tensor) else class_loss

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
- 实例损失: {il_val:.6f}
- 类别损失: {cl_val:.6f}
- 总损失: {loss.detach().item():.6f}
""")


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
