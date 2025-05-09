import torch
import torch.nn.functional as F
import gc

def compute_loss(
    accelerator, batch, unet, vae, noise_scheduler,
    instance_text_inputs, class_text_inputs, text_encoder, 
    prior_preservation_weight, device, 
    mixed_precision_dtype, unet_dtype,
    config=None, # config is used for train_text_encoder flag
    text_encoder_2=None # 明确添加第二个文本编码器作为参数
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

        # 检测各组件的设备情况
        vae_device = next(vae.parameters()).device
        unet_device = next(unet.parameters()).device
        
        # 输出日志显示设备不一致情况
        if vae_device != unet_device:
            accelerator.print(f"[警告] VAE设备({vae_device})与UNet设备({unet_device})不一致")
        
        # 获取UNet的数据类型以确保一致性
        unet_dtype_from_model = next(unet.parameters()).dtype
        if unet_dtype != unet_dtype_from_model:
            accelerator.print(f"[警告] 提供的UNet数据类型({unet_dtype})与模型参数类型({unet_dtype_from_model})不匹配，将使用模型参数类型")
            unet_dtype = unet_dtype_from_model
        
        # 使用VAE的设备为所有输入的目标设备
        target_device = vae_device
        accelerator.print(f"[INFO] 将使用VAE的设备({target_device})作为所有输入的目标设备")
        accelerator.print(f"[INFO] 使用UNet数据类型: {unet_dtype} 进行计算")
        
        # 将输入数据明确移动到正确的设备和数据类型
        pixel_values = batch["pixel_values"].to(device=target_device, dtype=torch.float32)  # VAE 始终使用 float32
        is_instance = batch["is_instance"].to(device=target_device)
        
        # 调试: 打印 is_instance 的具体值和设备
        if accelerator.is_main_process:
            accelerator.print(f"[DEBUG] is_instance values: {is_instance} 设备: {is_instance.device}")

        if pixel_values.shape[0] == 0:
            return torch.tensor(0.0, device=target_device, dtype=mixed_precision_dtype, requires_grad=False), 0.0, 0.0
        
        # 检测显存使用情况
        if torch.cuda.is_available() and accelerator.is_main_process and config["memory_optimization"].get("monitor_memory", False):
            current_mem = torch.cuda.memory_allocated() / 1024**3
            max_mem = torch.cuda.max_memory_allocated() / 1024**3
            accelerator.print(f"[内存监控] 损失计算开始: {current_mem:.2f}GB (峰值: {max_mem:.2f}GB)")

        # VAE 总是使用 float32，确保在VAE的设备上
        pixel_values_for_vae = pixel_values.to(device=target_device, dtype=torch.float32)
        
        # 强制确认VAE和输入在同一设备上
        accelerator.print(f"[DEBUG] VAE 设备: {vae_device}, 输入设备: {pixel_values_for_vae.device}")
        
        # 捕获具体的编码错误并尝试修复
        try:
            with torch.no_grad():
                latents = vae.encode(pixel_values_for_vae).latent_dist.sample()
                latents = latents * vae.config.scaling_factor
                # 在VAE编码后立即释放原始像素数据以节省内存
                del pixel_values_for_vae
                if config["memory_optimization"].get("aggressive_gc", False):
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
        except RuntimeError as e:
            error_msg = str(e)
            if "Input type" in error_msg and "weight type" in error_msg:
                accelerator.print(f"[错误] 设备不匹配问题: {error_msg}")
                accelerator.print("[尝试] 将VAE移动到与输入数据相同的设备上...")
                # 如果输入已经确定在CUDA上，尝试移动模型到同一设备
                if pixel_values_for_vae.is_cuda:
                    new_device = pixel_values_for_vae.device
                    accelerator.print(f"[修复] 将VAE移动到 {new_device}")
                    vae = vae.to(new_device)
                    # 重试编码
                    with torch.no_grad():
                        latents = vae.encode(pixel_values_for_vae).latent_dist.sample()
                        latents = latents * vae.config.scaling_factor
                else:
                    # 如果输入在CPU上，尝试将输入移到CUDA
                    accelerator.print(f"[修复] 将输入移动到 {vae_device}")
                    pixel_values_for_vae = pixel_values_for_vae.to(vae_device)
                    # 重试编码
                    with torch.no_grad():
                        latents = vae.encode(pixel_values_for_vae).latent_dist.sample()
                        latents = latents * vae.config.scaling_factor
            else:
                # 其他类型的错误，重新抛出
                raise
        
        # 确保latents在正确的设备和数据类型上
        latents = latents.to(device=target_device, dtype=unet_dtype)

        noise = torch.randn_like(latents)  # 这将继承 latents 的数据类型和设备
        
        # 确保时间步数据也匹配 UNet 参数的数据类型和设备
        timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, 
                                 (latents.shape[0],), device=target_device).long()
        
        # 确保所有时间步都是长整型，并且在正确的设备上
        timesteps = timesteps.to(device=target_device, dtype=torch.long)
        
        # 打印调试信息确认设备
        if accelerator.is_main_process:
            accelerator.print(f"[DEBUG] 时间步设备: {timesteps.device}, UNet设备: {unet_device}, VAE设备: {vae_device}")
            accelerator.print(f"[DEBUG] 噪声设备: {noise.device}")
            accelerator.print(f"[DEBUG] 潜变量设备: {latents.device}")
        
        # 将UNet移动到与其他组件相同的设备上
        if unet_device != target_device:
            accelerator.print(f"[修复] 将UNet移动到 {target_device}")
            unet = unet.to(target_device)
        
        noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
        
        instance_emb = None
        class_emb = None
        
        # 改进的SDXL模型检测
        is_sdxl = False
        if config and "advanced" in config and "model_type" in config["advanced"]:
            is_sdxl = config["advanced"]["model_type"].lower() == "sdxl"
        # 如果配置中没有指定，通过模型特征检测
        elif hasattr(unet, "config") and hasattr(unet.config, "cross_attention_dim"):
            is_sdxl = unet.config.cross_attention_dim >= 2048
        elif hasattr(unet, "add_embedding") and "added_cond_kwargs" in str(unet.forward.__code__.co_varnames):
            is_sdxl = True
        
        # 记录模型类型用于调试
        if accelerator.is_main_process:
            accelerator.print(f"[配置] 使用模型类型: {'SDXL' if is_sdxl else 'SD'}")
            if is_sdxl and text_encoder_2 is None:
                accelerator.print("[警告] 检测到SDXL模型但没有提供第二个文本编码器!")
        
        # 为SDXL模型准备额外的条件参数，确保在正确设备上
        added_cond_kwargs = None
        if is_sdxl:
            # 创建默认的text_embeds和time_ids
            batch_size = latents.shape[0]
            # 为SDXL模型创建text_embeds（维度为1280），明确指定设备和数据类型
            text_embeds = torch.zeros((batch_size, 1280), device=target_device, dtype=unet_dtype)
            # 为SDXL模型创建time_ids（使用默认形状）
            time_ids = torch.zeros((batch_size, 6), device=target_device, dtype=unet_dtype)
            added_cond_kwargs = {"text_embeds": text_embeds, "time_ids": time_ids}
            
            accelerator.print(f"[INFO] 已检测到SDXL模型，添加必要的条件参数")
            accelerator.print(f"[DEBUG] added_cond_kwargs['text_embeds'] 设备: {text_embeds.device}")
            accelerator.print(f"[DEBUG] added_cond_kwargs['time_ids'] 设备: {time_ids.device}")
        
        # 确保input_ids在正确的设备上
        if instance_text_inputs is not None:
            if instance_text_inputs.input_ids.device != target_device:
                instance_text_inputs.input_ids = instance_text_inputs.input_ids.to(target_device)
        
        if class_text_inputs is not None:
            if class_text_inputs.input_ids.device != target_device:
                class_text_inputs.input_ids = class_text_inputs.input_ids.to(target_device)
        
        # 更新弃用的autocast API
        text_encoder_precision_context = torch.amp.autocast(
            "cuda", 
            dtype=mixed_precision_dtype if config["memory_optimization"].get("lower_text_encoder_precision", False) else torch.float32
        ) if config["memory_optimization"].get("lower_text_encoder_precision", False) else torch.no_grad()
        
        # 处理实例文本嵌入
        if torch.any(is_instance):
            with text_encoder_precision_context:
                with torch.set_grad_enabled(config["training"]["train_text_encoder"]):
                    # 确保text_encoder在正确的设备上
                    if hasattr(text_encoder, 'device') and text_encoder.device != target_device:
                        accelerator.print(f"[警告] 文本编码器不在正确的设备上，尝试移动到 {target_device}")
                        text_encoder = text_encoder.to(target_device)
                    
                    instance_emb_raw = text_encoder(instance_text_inputs.input_ids)[0]
                    
                    # SDXL模型需要特殊处理文本嵌入
                    if is_sdxl and text_encoder_2 is not None:
                        # 确保text_encoder_2在正确的设备上
                        if hasattr(text_encoder_2, 'device') and text_encoder_2.device != target_device:
                            accelerator.print(f"[警告] 第二文本编码器不在正确的设备上，尝试移动到 {target_device}")
                            text_encoder_2 = text_encoder_2.to(target_device)
                        
                        # 确保二次编码器的输入在正确的设备上
                        instance_emb_raw_2 = text_encoder_2(instance_text_inputs.input_ids.to(text_encoder_2.device))[0]
                        
                        # 打印调试信息以了解张量形状
                        accelerator.print(f"[DEBUG] 文本编码器输出形状: {instance_emb_raw.shape}, 类型: {instance_emb_raw.dtype}")
                        accelerator.print(f"[DEBUG] 文本编码器2输出形状: {instance_emb_raw_2.shape}, 类型: {instance_emb_raw_2.dtype}")
                        
                        # 处理可能的维度不匹配
                        if len(instance_emb_raw.shape) == 2:
                            accelerator.print("[INFO] 将文本编码器1输出从2D转换为3D")
                            batch_size, hidden_dim = instance_emb_raw.shape
                            instance_emb_raw = instance_emb_raw.unsqueeze(1)
                        
                        if len(instance_emb_raw_2.shape) == 2:
                            accelerator.print("[INFO] 将文本编码器2输出从2D转换为3D")
                            batch_size, hidden_dim = instance_emb_raw_2.shape
                            instance_emb_raw_2 = instance_emb_raw_2.unsqueeze(1)
                        
                        if instance_emb_raw.shape[1] != instance_emb_raw_2.shape[1]:
                            seq_len_1 = instance_emb_raw.shape[1]
                            seq_len_2 = instance_emb_raw_2.shape[1]
                            max_seq_len = max(seq_len_1, seq_len_2)
                            accelerator.print(f"[INFO] 序列长度不同: encoder1={seq_len_1}, encoder2={seq_len_2}, 统一到{max_seq_len}")
                            
                            if seq_len_1 < max_seq_len:
                                pad_len = max_seq_len - seq_len_1
                                padding = torch.zeros((instance_emb_raw.shape[0], pad_len, instance_emb_raw.shape[2]), 
                                                    device=target_device, dtype=instance_emb_raw.dtype)
                                instance_emb_raw = torch.cat([instance_emb_raw, padding], dim=1)
                            
                            if seq_len_2 < max_seq_len:
                                pad_len = max_seq_len - seq_len_2
                                padding = torch.zeros((instance_emb_raw_2.shape[0], pad_len, instance_emb_raw_2.shape[2]), 
                                                    device=target_device, dtype=instance_emb_raw_2.dtype)
                                instance_emb_raw_2 = torch.cat([instance_emb_raw_2, padding], dim=1)
                        
                        try:
                            instance_emb_raw = torch.cat([
                                instance_emb_raw.to(target_device),
                                instance_emb_raw_2.to(target_device)
                            ], dim=-1)
                            
                            accelerator.print(f"[INFO] 成功连接文本编码器输出，最终形状: {instance_emb_raw.shape}")
                        except Exception as e:
                            accelerator.print(f"[ERROR] 连接文本编码器输出时出错: {e}")
                            accelerator.print("[WARN] 回退到仅使用第一个文本编码器的输出")
                
            # 确保实例嵌入的数据类型与UNet匹配
            instance_emb = instance_emb_raw.to(device=target_device, dtype=unet_dtype)
            
            # 立即删除原始张量以节省内存
            del instance_emb_raw
            if is_sdxl and text_encoder_2 is not None and "instance_emb_raw_2" in locals():
                del instance_emb_raw_2
                
            # 在每次主要操作后执行内存回收
            if config["memory_optimization"].get("aggressive_gc", False):
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        # 处理类别文本嵌入（使用相同的优化）
        if class_text_inputs is not None and torch.any(~is_instance) and prior_preservation_weight > 0:
            with text_encoder_precision_context:
                with torch.set_grad_enabled(config["training"]["train_text_encoder"]):
                    class_emb_raw = text_encoder(class_text_inputs.input_ids.to(target_device))[0]
                    
                    if is_sdxl and text_encoder_2 is not None:
                        class_emb_raw_2 = text_encoder_2(class_text_inputs.input_ids.to(text_encoder_2.device))[0]
                        
                        if len(class_emb_raw.shape) == 2:
                            batch_size, hidden_dim = class_emb_raw.shape
                            class_emb_raw = class_emb_raw.unsqueeze(1)
                        
                        if len(class_emb_raw_2.shape) == 2:
                            batch_size, hidden_dim = class_emb_raw_2.shape
                            class_emb_raw_2 = class_emb_raw_2.unsqueeze(1)
                        
                        if class_emb_raw.shape[1] != class_emb_raw_2.shape[1]:
                            seq_len_1 = class_emb_raw.shape[1]
                            seq_len_2 = class_emb_raw_2.shape[1]
                            max_seq_len = max(seq_len_1, seq_len_2)
                            
                            if seq_len_1 < max_seq_len:
                                pad_len = max_seq_len - seq_len_1
                                padding = torch.zeros((class_emb_raw.shape[0], pad_len, class_emb_raw.shape[2]), 
                                                    device=target_device, dtype=class_emb_raw.dtype)
                                class_emb_raw = torch.cat([class_emb_raw, padding], dim=1)
                            
                            if seq_len_2 < max_seq_len:
                                pad_len = max_seq_len - seq_len_2
                                padding = torch.zeros((class_emb_raw_2.shape[0], pad_len, class_emb_raw_2.shape[2]), 
                                                    device=target_device, dtype=class_emb_raw_2.dtype)
                                class_emb_raw_2 = torch.cat([class_emb_raw_2, padding], dim=1)
                        
                        try:
                            class_emb_raw = torch.cat([
                                class_emb_raw.to(target_device),
                                class_emb_raw_2.to(target_device)
                            ], dim=-1)
                        except Exception as e:
                            accelerator.print(f"[ERROR] 连接类别文本编码器输出时出错: {e}")
                            accelerator.print("[WARN] 回退到仅使用第一个文本编码器的类别输出")
                
            # 确保类别嵌入的数据类型与UNet匹配
            class_emb = class_emb_raw.to(device=target_device, dtype=unet_dtype)
                
            # 立即删除类别嵌入原始数据以节省内存
            del class_emb_raw
            if is_sdxl and text_encoder_2 is not None and "class_emb_raw_2" in locals():
                del class_emb_raw_2
            
            # 增加周期性内存清理
            if config["memory_optimization"].get("aggressive_gc", False) and torch.cuda.is_available():
                gc.collect()
                torch.cuda.empty_cache()
                # 重置峰值内存统计
                torch.cuda.reset_peak_memory_stats()

        # 计算损失时使用混合精度来减少显存使用
        loss_precision_context = torch.amp.autocast(
            "cuda", 
            dtype=torch.float32  # 损失计算通常使用float32以确保稳定性
        ) if config["memory_optimization"].get("lower_loss_precision", False) else torch.no_grad()
        
        instance_loss_val = 0.0
        class_loss_val = 0.0
        
        instance_count = torch.sum(is_instance).item()
        class_count = torch.sum(~is_instance).item()

        if instance_count > 0:
            # 为instance准备SDXL条件参数（如果需要）
            instance_added_cond_kwargs = None
            if is_sdxl and added_cond_kwargs:
                instance_added_cond_kwargs = {
                    "text_embeds": added_cond_kwargs["text_embeds"][is_instance].to(device=target_device, dtype=unet_dtype),
                    "time_ids": added_cond_kwargs["time_ids"][is_instance].to(device=target_device, dtype=unet_dtype)
                }
            
            # 确保所有输入都在正确的设备和数据类型上
            instance_input = noisy_latents[is_instance].to(device=target_device, dtype=unet_dtype)
            instance_timesteps = timesteps[is_instance].to(device=target_device, dtype=torch.long)
            instance_encoder_hidden = instance_emb.repeat(instance_count, 1, 1).to(device=target_device, dtype=unet_dtype)
            
            # 打印所有输入的数据类型和设备，帮助调试
            accelerator.print(f"[DEBUG] instance_input 类型: {instance_input.dtype}, 设备: {instance_input.device}")
            accelerator.print(f"[DEBUG] instance_encoder_hidden 类型: {instance_encoder_hidden.dtype}, 设备: {instance_encoder_hidden.device}")
            if instance_added_cond_kwargs:
                for key, value in instance_added_cond_kwargs.items():
                    accelerator.print(f"[DEBUG] instance_added_cond_kwargs[{key}] 类型: {value.dtype}, 设备: {value.device}")
            
            # 确保 timesteps 的类型是 long，而不是 UNet 期望的其他类型
            try:
                instance_pred = unet(
                    instance_input,
                    instance_timesteps,
                    encoder_hidden_states=instance_encoder_hidden,
                    added_cond_kwargs=instance_added_cond_kwargs
                ).sample
            except RuntimeError as e:
                error_msg = str(e)
                accelerator.print(f"UNet 实例预测出错: {error_msg}")
                
                # 针对数据类型不匹配问题的具体处理
                if "must have the same dtype" in error_msg:
                    accelerator.print("[修复] 尝试统一所有输入张量的数据类型...")
                    
                    # 尝试使用全精度Float32来避免混合精度问题
                    instance_input = instance_input.float()
                    instance_encoder_hidden = instance_encoder_hidden.float()
                    
                    if instance_added_cond_kwargs:
                        for key in instance_added_cond_kwargs:
                            instance_added_cond_kwargs[key] = instance_added_cond_kwargs[key].float()
                    
                    # 使用统一的Float32数据类型重试
                    accelerator.print("[尝试] 使用Float32数据类型重新运行UNet...")
                    instance_pred = unet(
                        instance_input,
                        instance_timesteps,
                        encoder_hidden_states=instance_encoder_hidden,
                        added_cond_kwargs=instance_added_cond_kwargs
                    ).sample
                else:
                    # 对于其他类型的错误，尝试不使用added_cond_kwargs
                    accelerator.print("尝试不使用added_cond_kwargs...")
                    instance_pred = unet(
                        instance_input,
                        instance_timesteps,
                        encoder_hidden_states=instance_encoder_hidden
                    ).sample
            
            # MSE 损失计算应该使用 float32 以提高数值稳定性
            instance_loss_tensor = F.mse_loss(
                instance_pred.float(),  # 转换为 float32 用于计算损失
                noise[is_instance].to(target_device).float(),
                reduction="mean"
            )
            instance_loss_val = instance_loss_tensor
            
            # 强制删除大型张量，减少内存占用
            del instance_pred
            # 无需在这里保留噪声张量的实例部分
        
        if class_count > 0:
            # 为class准备SDXL条件参数（如果需要）
            class_added_cond_kwargs = None
            if is_sdxl and added_cond_kwargs:
                class_added_cond_kwargs = {
                    "text_embeds": added_cond_kwargs["text_embeds"][~is_instance].to(device=target_device, dtype=unet_dtype),
                    "time_ids": added_cond_kwargs["time_ids"][~is_instance].to(device=target_device, dtype=unet_dtype)
                }
            
            # 确保所有输入都在正确的设备和数据类型上
            class_input = noisy_latents[~is_instance].to(device=target_device, dtype=unet_dtype)
            class_timesteps = timesteps[~is_instance].to(device=target_device, dtype=torch.long)
            class_encoder_hidden = class_emb.repeat(class_count, 1, 1).to(device=target_device, dtype=unet_dtype)
            
            # 打印确认所有输入的数据类型和设备
            accelerator.print(f"[DEBUG] class_input 类型: {class_input.dtype}, 设备: {class_input.device}")
            accelerator.print(f"[DEBUG] class_encoder_hidden 类型: {class_encoder_hidden.dtype}, 设备: {class_encoder_hidden.device}")
            if class_added_cond_kwargs:
                for key, value in class_added_cond_kwargs.items():
                    accelerator.print(f"[DEBUG] class_added_cond_kwargs[{key}] 类型: {value.dtype}, 设备: {value.device}")
            
            # 确保所有输入到 unet 的数据都是正确的数据类型
            try:
                class_pred = unet(
                    class_input,
                    class_timesteps,
                    encoder_hidden_states=class_encoder_hidden,
                    added_cond_kwargs=class_added_cond_kwargs
                ).sample
            except RuntimeError as e:
                error_msg = str(e)
                accelerator.print(f"UNet 类别预测出错: {error_msg}")
                
                # 针对数据类型不匹配问题的具体处理
                if "must have the same dtype" in error_msg:
                    accelerator.print("[修复] 尝试统一所有输入张量的数据类型...")
                    
                    # 尝试使用全精度Float32来避免混合精度问题
                    class_input = class_input.float()
                    class_encoder_hidden = class_encoder_hidden.float()
                    
                    if class_added_cond_kwargs:
                        for key in class_added_cond_kwargs:
                            class_added_cond_kwargs[key] = class_added_cond_kwargs[key].float()
                    
                    # 使用统一的Float32数据类型重试
                    accelerator.print("[尝试] 使用Float32数据类型重新运行UNet...")
                    class_pred = unet(
                        class_input,
                        class_timesteps,
                        encoder_hidden_states=class_encoder_hidden,
                        added_cond_kwargs=class_added_cond_kwargs
                    ).sample
                else:
                    # 尝试不使用added_cond_kwargs
                    accelerator.print("尝试不使用added_cond_kwargs...")
                    class_pred = unet(
                        class_input,
                        class_timesteps,
                        encoder_hidden_states=class_encoder_hidden
                    ).sample
            
            # MSE 损失计算应该使用 float32 以提高数值稳定性
            class_loss_tensor = F.mse_loss(
                class_pred.float(),  # 转换为 float32 用于计算损失
                noise[~is_instance].to(target_device).float(),
                reduction="mean"
            )
            class_loss_val = class_loss_tensor
            
            # 强制删除大型张量，减少内存占用
            del class_pred
        
        # 在每次迭代结束时显式释放未使用的张量
        if config["memory_optimization"].get("aggressive_gc", False) and torch.cuda.is_available():
            # 显式删除大型中间结果
            del noise
            del noisy_latents
            if "instance_emb" in locals() and instance_emb is not None:
                del instance_emb
            if "class_emb" in locals() and class_emb is not None:
                del class_emb
            
            # 释放CUDA缓存
            torch.cuda.empty_cache()
            
            # 监控内存使用
            if accelerator.is_main_process and config["memory_optimization"].get("monitor_memory", False):
                current_mem = torch.cuda.memory_allocated() / 1024**3
                max_mem = torch.cuda.max_memory_allocated() / 1024**3
                accelerator.print(f"[内存监控] 损失计算结束: {current_mem:.2f}GB (峰值: {max_mem:.2f}GB)")
        
        if isinstance(instance_loss_val, torch.Tensor) and isinstance(class_loss_val, torch.Tensor):
            loss = instance_loss_val + prior_preservation_weight * class_loss_val
        elif isinstance(instance_loss_val, torch.Tensor):
            loss = instance_loss_val
        elif isinstance(class_loss_val, torch.Tensor):
            loss = prior_preservation_weight * class_loss_val
        else: # Both are 0.0
            loss = torch.tensor(0.0, device=target_device, dtype=mixed_precision_dtype) # Ensure it's a tensor
        
        return loss, instance_loss_val, class_loss_val
    
    except Exception as e:
        accelerator.print(f"损失计算错误: {e}")
        import traceback
        accelerator.print(traceback.format_exc())
        # Ensure a tensor is returned for loss, even in error, to prevent crashes upstream
        error_loss = torch.tensor(float('inf'), device=target_device, dtype=torch.float32 if not isinstance(mixed_precision_dtype, str) else torch.float32) # Fallback to float32
        
        # 发生错误时，强制进行内存清理
        if torch.cuda.is_available():
            try:
                # 清理所有局部变量中的张量
                local_vars = locals()
                tensor_vars = [var for var in local_vars if isinstance(local_vars[var], torch.Tensor)]
                for var in tensor_vars:
                    del local_vars[var]
                
                # 彻底清理内存
                gc.collect()
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                accelerator.print("[内存清理] 已执行紧急内存清理")
            except:
                pass
        
        return error_loss, 0.0, 0.0
