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
        
        # 检查是否使用SDXL模型 - 更健壮的检测方法
        is_sdxl = False
        # 首先从配置中检查
        if config and "advanced" in config and "model_type" in config["advanced"]:
            is_sdxl = config["advanced"]["model_type"].lower() == "sdxl"
        else:
            # 从UNet的特性检查是否为SDXL
            # SDXL的UNet通常具有更大的cross_attention_dim (2048 vs 768/1024)
            if hasattr(unet, "config") and hasattr(unet.config, "cross_attention_dim"):
                is_sdxl = unet.config.cross_attention_dim >= 2048
            elif hasattr(unet, "add_embedding") and "added_cond_kwargs" in str(unet.forward.__code__.co_varnames):
                is_sdxl = True
        
        # 为SDXL模型准备额外的条件参数
        added_cond_kwargs = None
        if is_sdxl:
            # 创建默认的text_embeds和time_ids
            batch_size = latents.shape[0]
            # 为SDXL模型创建text_embeds（维度为1280）
            text_embeds = torch.zeros((batch_size, 1280), device=device, dtype=unet_dtype)
            # 为SDXL模型创建time_ids（使用默认形状）
            time_ids = torch.zeros((batch_size, 6), device=device, dtype=unet_dtype)
            added_cond_kwargs = {"text_embeds": text_embeds, "time_ids": time_ids}
            
            accelerator.print(f"[INFO] 已检测到SDXL模型，添加必要的条件参数")
        
        # 对于SDXL和非SDXL模型使用不同的文本处理方法
        if torch.any(is_instance):
            with torch.set_grad_enabled(config["training"]["train_text_encoder"]):
                instance_emb_raw = text_encoder(instance_text_inputs.input_ids)[0]
                
                # SDXL模型需要特殊处理文本嵌入
                if is_sdxl:
                    # 检查是否有第二个文本编码器
                    if "text_encoder_2" in config and config["text_encoder_2"] is not None:
                        text_encoder_2 = config["text_encoder_2"]
                        instance_emb_raw_2 = text_encoder_2(instance_text_inputs.input_ids)[0]
                        # 创建SDXL特定的嵌入格式
                        # 注意：SDXL的UNet期望的嵌入维度为77×2048
                        # 我们需要将两个编码器的输出重新调整为正确的尺寸
                        instance_emb_raw = torch.cat([
                            torch.zeros((instance_emb_raw.shape[0], 77, 2048-768-1280), 
                                       device=device, dtype=instance_emb_raw.dtype),
                            instance_emb_raw,  # 第一个编码器的输出 (77x768)
                            instance_emb_raw_2  # 第二个编码器的输出 (77x1280) 
                        ], dim=-1)
                    else:
                        # 如果没有第二个编码器，进行简单填充
                        accelerator.print("[WARN] SDXL模型但缺少第二个文本编码器，进行零填充")
                        # 填充到SDXL期望的2048维度
                        padding = torch.zeros((instance_emb_raw.shape[0], 77, 2048-768), 
                                             device=device, dtype=instance_emb_raw.dtype)
                        instance_emb_raw = torch.cat([instance_emb_raw, padding], dim=-1)
                
            # 确保它与 unet 数据类型匹配
            instance_emb = instance_emb_raw.to(dtype=unet_dtype)

        if class_text_inputs is not None and torch.any(~is_instance) and prior_preservation_weight > 0:
            with torch.set_grad_enabled(config["training"]["train_text_encoder"]):
                class_emb_raw = text_encoder(class_text_inputs.input_ids)[0]
                
                # SDXL模型需要特殊处理文本嵌入
                if is_sdxl:
                    # 检查是否有第二个文本编码器
                    if "text_encoder_2" in config and config["text_encoder_2"] is not None:
                        text_encoder_2 = config["text_encoder_2"]
                        class_emb_raw_2 = text_encoder_2(class_text_inputs.input_ids)[0]
                        # 创建SDXL特定的嵌入格式
                        class_emb_raw = torch.cat([
                            torch.zeros((class_emb_raw.shape[0], 77, 2048-768-1280), 
                                       device=device, dtype=class_emb_raw.dtype),
                            class_emb_raw,  # 第一个编码器 (77x768)
                            class_emb_raw_2  # 第二个编码器 (77x1280)
                        ], dim=-1)
                    else:
                        # 如果没有第二个编码器，进行简单填充
                        # 填充到SDXL期望的2048维度
                        padding = torch.zeros((class_emb_raw.shape[0], 77, 2048-768), 
                                             device=device, dtype=class_emb_raw.dtype)
                        class_emb_raw = torch.cat([class_emb_raw, padding], dim=-1)
                
            # 确保它与 unet 数据类型匹配
            class_emb = class_emb_raw.to(dtype=unet_dtype)

        instance_loss_val = 0.0
        class_loss_val = 0.0
        
        instance_count = torch.sum(is_instance).item()
        class_count = torch.sum(~is_instance).item()

        if instance_count > 0:
            # 为instance准备SDXL条件参数（如果需要）
            instance_added_cond_kwargs = None
            if is_sdxl and added_cond_kwargs:
                instance_added_cond_kwargs = {
                    "text_embeds": added_cond_kwargs["text_embeds"][is_instance],
                    "time_ids": added_cond_kwargs["time_ids"][is_instance]
                }
            
            # 确保 timesteps 的类型是 long，而不是 UNet 期望的其他类型
            instance_pred = unet(
                noisy_latents[is_instance],
                timesteps[is_instance],
                encoder_hidden_states=instance_emb.repeat(instance_count, 1, 1),
                added_cond_kwargs=instance_added_cond_kwargs
            ).sample
            
            # MSE 损失计算应该使用 float32 以提高数值稳定性
            instance_loss_tensor = F.mse_loss(
                instance_pred.float(),  # 转换为 float32 用于计算损失
                noise[is_instance].float(),
                reduction="mean"
            )
            instance_loss_val = instance_loss_tensor
        
        if class_count > 0:
            # 为class准备SDXL条件参数（如果需要）
            class_added_cond_kwargs = None
            if is_sdxl and added_cond_kwargs:
                class_added_cond_kwargs = {
                    "text_embeds": added_cond_kwargs["text_embeds"][~is_instance],
                    "time_ids": added_cond_kwargs["time_ids"][~is_instance]
                }
                
            # 确保所有输入到 unet 的数据都是正确的数据类型
            class_pred = unet(
                noisy_latents[~is_instance].to(dtype=unet_dtype),  # 显式转换为与 unet 相同的数据类型
                timesteps[~is_instance].long(),  # 确保时间步是长整数
                encoder_hidden_states=class_emb.repeat(class_count, 1, 1).to(dtype=unet_dtype),  # 显式转换数据类型
                added_cond_kwargs=class_added_cond_kwargs
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
