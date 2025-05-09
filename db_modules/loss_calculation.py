import torch
import torch.nn.functional as F

def compute_loss(
    accelerator, batch, unet, vae, noise_scheduler,
    instance_text_inputs, class_text_inputs, text_encoder, 
    prior_preservation_weight, device, 
    mixed_precision_dtype, unet_dtype,
    config=None # config is used for train_text_encoder flag
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

        instance_loss_val = 0.0
        class_loss_val = 0.0
        
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
            instance_loss_val = instance_loss_tensor
        
        if class_count > 0:
            # 确保所有输入到 unet 的数据都是正确的数据类型
            class_pred = unet(
                noisy_latents[~is_instance].to(dtype=unet_dtype),  # 显式转换为与 unet 相同的数据类型
                timesteps[~is_instance].long(),  # 确保时间步是长整数
                encoder_hidden_states=class_emb.repeat(class_count, 1, 1).to(dtype=unet_dtype)  # 显式转换数据类型
            ).sample
            
            # MSE 损失计算应该使用 float32 以提高数值稳定性
            class_loss_tensor = F.mse_loss(
                class_pred.float(),  # 转换为 float32 用于计算损失
                noise[~is_instance].float(),
                reduction="mean"
            )
            class_loss_val = class_loss_tensor
        
        if isinstance(instance_loss_val, torch.Tensor) and isinstance(class_loss_val, torch.Tensor):
            loss = instance_loss_val + prior_preservation_weight * class_loss_val
        elif isinstance(instance_loss_val, torch.Tensor):
            loss = instance_loss_val
        elif isinstance(class_loss_val, torch.Tensor):
            loss = prior_preservation_weight * class_loss_val
        else: # Both are 0.0
            loss = torch.tensor(0.0, device=device, dtype=mixed_precision_dtype) # Ensure it's a tensor
        
        return loss, instance_loss_val, class_loss_val
    
    except Exception as e:
        accelerator.print(f"损失计算错误: {e}")
        import traceback
        accelerator.print(traceback.format_exc())
        # Ensure a tensor is returned for loss, even in error, to prevent crashes upstream
        error_loss = torch.tensor(float('inf'), device=device, dtype=torch.float32 if not isinstance(mixed_precision_dtype, str) else torch.float32) # Fallback to float32
        return error_loss, 0.0, 0.0
