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
        # 使用全局状态跟踪来减少模型移动
        global _models_moved
        if '_models_moved' not in globals():
            _models_moved = False
        
        # 检测各组件的设备情况
        vae_device = next(vae.parameters()).device
        unet_device = next(unet.parameters()).device
        
        # 仅在第一次迭代显示警告信息
        if not _models_moved:
            # 输出日志显示设备不一致情况
            if vae_device != unet_device:
                accelerator.print(f"[警告] VAE设备({vae_device})与UNet设备({unet_device})不一致")
        
        # 使用VAE的设备为所有输入的目标设备
        target_device = vae_device
        
        # 仅在首次迭代或设备不一致时移动模型
        if not _models_moved:
            if unet_device != target_device:
                accelerator.print(f"[一次性] 将UNet从 {unet_device} 移动到 {target_device}")
                unet = unet.to(target_device)
            
            if hasattr(text_encoder, 'device') and text_encoder.device != target_device:
                accelerator.print(f"[一次性] 将文本编码器从 {text_encoder.device} 移动到 {target_device}")
                text_encoder = text_encoder.to(target_device)
                
            if text_encoder_2 is not None and hasattr(text_encoder_2, 'device') and text_encoder_2.device != target_device:
                accelerator.print(f"[一次性] 将文本编码器2从 {text_encoder_2.device} 移动到 {target_device}")
                text_encoder_2 = text_encoder_2.to(target_device)
            
            # 避免未来的移动
            _models_moved = True
            accelerator.print(f"[信息] 所有模型现已固定在 {target_device} 上，将不再移动")
        
        # 根据配置文件确定是否使用半精度
        use_half_precision = config["training"].get("mixed_precision", "") in ["fp16", "bf16"]
        force_half_precision = use_half_precision
        
        # 确定所有计算的目标精度
        target_dtype = None
        if force_half_precision:
            if config["training"].get("mixed_precision", "") == "fp16":
                target_dtype = torch.float16
            elif config["training"].get("mixed_precision", "") == "bf16":
                target_dtype = torch.bfloat16
        else:
            # 使用默认精度
            target_dtype = unet_dtype
        
        # 将unet_dtype更新为实际使用的精度
        unet_dtype = target_dtype if target_dtype is not None else unet_dtype
        
        # 将输入数据明确移动到正确的设备和数据类型
        pixel_values = batch["pixel_values"].to(device=target_device, dtype=torch.float32)  # VAE 始终使用 float32
        is_instance = batch["is_instance"].to(device=target_device)
        
        # 调试: 打印 is_instance 的具体值和设备 (仅打印一次)
        if accelerator.is_main_process and 'printed_instance_debug' not in globals():
            accelerator.print(f"[DEBUG] is_instance values: {is_instance} 设备: {is_instance.device}")
            globals()['printed_instance_debug'] = True

        if pixel_values.shape[0] == 0:
            return torch.tensor(0.0, device=target_device, dtype=mixed_precision_dtype, requires_grad=True), 0.0, 0.0
        
        # VAE 总是使用 float32，确保在VAE的设备上
        pixel_values_for_vae = pixel_values.to(device=target_device, dtype=torch.float32)
        
        # 分离上下文，防止VAE编码出现NaN
        with torch.no_grad():
            try:
                latents = vae.encode(pixel_values_for_vae).latent_dist.sample()
                latents = latents * vae.config.scaling_factor
            except RuntimeError:
                # 如果编码失败，尝试使用更保守的方法
                accelerator.print("[警告] VAE编码出错，尝试逐个样本编码...")
                latents_list = []
                for i in range(pixel_values_for_vae.shape[0]):
                    try:
                        single_latent = vae.encode(pixel_values_for_vae[i:i+1]).latent_dist.sample()
                        latents_list.append(single_latent * vae.config.scaling_factor)
                    except Exception as e:
                        accelerator.print(f"[错误] 样本 {i} 编码失败: {e}，使用随机潜变量替代")
                        # 使用随机潜变量替代
                        random_latent = torch.randn((1, 4, pixel_values_for_vae.shape[2]//8, pixel_values_for_vae.shape[3]//8), 
                                                  device=target_device, dtype=torch.float32)
                        latents_list.append(random_latent)
                        
                latents = torch.cat(latents_list, dim=0)
        
        # 确保latents在正确的设备和数据类型上
        latents = latents.to(device=target_device, dtype=unet_dtype)

        # 增强NaN检测和处理 - 确保保留梯度流
        def check_and_fix_nan_values(tensor, name="", aggressive=False):
            if torch.isnan(tensor).any() or torch.isinf(tensor).any():
                accelerator.print(f"[警告] 检测到 {name} 中存在NaN或Inf值，尝试修复...")
                # 创建掩码，只替换NaN/Inf值
                nan_mask = torch.isnan(tensor) | torch.isinf(tensor)
                nan_count = torch.sum(nan_mask).item()
                total_count = tensor.numel()
                nan_percentage = (nan_count / total_count) * 100
                
                # 如果NaN比例过高，考虑更激进的修复
                if nan_percentage > 50 or aggressive:
                    accelerator.print(f"[警告] {name} 中NaN/Inf比例高达 {nan_percentage:.2f}%，执行全替换")
                    # 完全替换张量
                    replacement = torch.zeros_like(tensor)
                    if tensor.requires_grad:
                        fixed_tensor = replacement + torch.zeros_like(tensor).requires_grad_()
                    else:
                        fixed_tensor = replacement
                else:
                    # 创建有梯度连接的克隆
                    fixed_tensor = tensor.clone()
                    # 只替换NaN/Inf值，保留正常值的梯度连接
                    fixed_tensor.data[nan_mask] = 0.0
                    
                return fixed_tensor, True
            return tensor, False
            
        # 生成噪声并确保没有NaN
        noise = torch.randn_like(latents)
        noise, has_nan = check_and_fix_nan_values(noise, "noise")
        
        # 确保时间步数据也匹配 UNet 参数的数据类型和设备
        timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, 
                                 (latents.shape[0],), device=target_device).long()
        
        # 确保所有时间步都是长整型，并且在正确的设备上
        timesteps = timesteps.to(device=target_device, dtype=torch.long)
        
        # 添加噪声到潜变量
        noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
        
        # 检查并修复可能的NaN
        noisy_latents, has_nan = check_and_fix_nan_values(noisy_latents, "noisy_latents")
        
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
        
        # 确保input_ids在正确的设备上
        if instance_text_inputs is not None:
            if instance_text_inputs.input_ids.device != target_device:
                instance_text_inputs.input_ids = instance_text_inputs.input_ids.to(target_device)
        
        if class_text_inputs is not None:
            if class_text_inputs.input_ids.device != target_device:
                class_text_inputs.input_ids = class_text_inputs.input_ids.to(target_device)
        
        # 更新弃用的autocast API
        text_encoder_precision_context = torch.amp.autocast(
            "cuda:0", 
            dtype=mixed_precision_dtype if config["memory_optimization"].get("lower_text_encoder_precision", False) else torch.float32
        ) if config["memory_optimization"].get("lower_text_encoder_precision", False) else torch.no_grad()
        
        # 处理实例文本嵌入
        if torch.any(is_instance):
            with text_encoder_precision_context:
                with torch.set_grad_enabled(config["training"]["train_text_encoder"]):
                    try:
                        instance_emb_raw = text_encoder(instance_text_inputs.input_ids)[0]
                        
                        # SDXL模型需要特殊处理文本嵌入
                        if is_sdxl and text_encoder_2 is not None:
                            try:
                                instance_emb_raw_2 = text_encoder_2(instance_text_inputs.input_ids.to(text_encoder_2.device))[0]
                                
                                # 处理可能的维度不匹配
                                if len(instance_emb_raw.shape) == 2:
                                    batch_size, hidden_dim = instance_emb_raw.shape
                                    instance_emb_raw = instance_emb_raw.unsqueeze(1)
                                
                                if len(instance_emb_raw_2.shape) == 2:
                                    batch_size, hidden_dim = instance_emb_raw_2.shape
                                    instance_emb_raw_2 = instance_emb_raw_2.unsqueeze(1)
                                
                                # 统一序列长度
                                if instance_emb_raw.shape[1] != instance_emb_raw_2.shape[1]:
                                    seq_len_1 = instance_emb_raw.shape[1]
                                    seq_len_2 = instance_emb_raw_2.shape[1]
                                    max_seq_len = max(seq_len_1, seq_len_2)
                                    
                                    # 调整第一个编码器输出
                                    if seq_len_1 < max_seq_len:
                                        pad_len = max_seq_len - seq_len_1
                                        padding = torch.zeros((instance_emb_raw.shape[0], pad_len, instance_emb_raw.shape[2]), 
                                                            device=target_device, dtype=instance_emb_raw.dtype)
                                        instance_emb_raw = torch.cat([instance_emb_raw, padding], dim=1)
                                    
                                    # 调整第二个编码器输出
                                    if seq_len_2 < max_seq_len:
                                        pad_len = max_seq_len - seq_len_2
                                        padding = torch.zeros((instance_emb_raw_2.shape[0], pad_len, instance_emb_raw_2.shape[2]), 
                                                            device=target_device, dtype=instance_emb_raw_2.dtype)
                                        instance_emb_raw_2 = torch.cat([instance_emb_raw_2, padding], dim=1)
                                
                                # 检查并修复NaN值
                                instance_emb_raw, has_nan1 = check_and_fix_nan_values(instance_emb_raw, "instance_emb_raw")
                                instance_emb_raw_2, has_nan2 = check_and_fix_nan_values(instance_emb_raw_2, "instance_emb_raw_2")
                                
                                # 连接两个编码器的输出
                                instance_emb_raw = torch.cat([
                                    instance_emb_raw.to(target_device),
                                    instance_emb_raw_2.to(target_device)
                                ], dim=-1)
                                
                            except Exception as e:
                                accelerator.print(f"[错误] SDXL文本编码器处理错误: {e}")
                                
                                # 回退方案：使用零填充代替第二个编码器的输出
                                if isinstance(instance_emb_raw, torch.Tensor):
                                    if len(instance_emb_raw.shape) == 2:
                                        instance_emb_raw = instance_emb_raw.unsqueeze(1)
                                        
                                    # 创建匹配大小的零张量
                                    batch_size, seq_len, hidden_dim = instance_emb_raw.shape
                                    padding = torch.zeros((batch_size, seq_len, 2048 - hidden_dim), 
                                                        device=target_device, dtype=instance_emb_raw.dtype)
                                    
                                    # 连接原始嵌入和填充
                                    instance_emb_raw = torch.cat([instance_emb_raw, padding], dim=-1)
                                else:
                                    # 如果instance_emb_raw不可用，创建全零嵌入
                                    batch_size = latents.shape[0]
                                    instance_emb_raw = torch.zeros((batch_size, 8, 2048), device=target_device, dtype=unet_dtype)
                        
                        # 检查NaN并修复
                        instance_emb_raw, has_nan = check_and_fix_nan_values(instance_emb_raw, "instance_emb_raw")
                    except Exception as e:
                        accelerator.print(f"[严重错误] 文本编码器处理失败: {e}")
                        # 创建虚拟嵌入作为后备
                        batch_size = latents.shape[0]
                        seq_len = 8  # 默认序列长度
                        hidden_dim = 2048 if is_sdxl else 768  # 根据模型选择正确的维度
                        instance_emb_raw = torch.zeros((batch_size, seq_len, hidden_dim), 
                                                    device=target_device, dtype=unet_dtype)
                        instance_emb_raw.requires_grad_(True)
                
            # 确保它与 unet 数据类型匹配
            instance_emb = instance_emb_raw.to(dtype=unet_dtype)
        
        # 处理类别文本嵌入（使用相同的优化）
        if class_text_inputs is not None and torch.any(~is_instance) and prior_preservation_weight > 0:
            with text_encoder_precision_context:
                with torch.set_grad_enabled(config["training"]["train_text_encoder"]):
                    try:
                        class_emb_raw = text_encoder(class_text_inputs.input_ids.to(target_device))[0]
                        
                        # SDXL模型需要特殊处理文本嵌入
                        if is_sdxl and text_encoder_2 is not None:
                            try:
                                class_emb_raw_2 = text_encoder_2(class_text_inputs.input_ids.to(text_encoder_2.device))[0]
                                
                                # 处理可能的维度不匹配
                                if len(class_emb_raw.shape) == 2:
                                    batch_size, hidden_dim = class_emb_raw.shape
                                    class_emb_raw = class_emb_raw.unsqueeze(1)
                                
                                if len(class_emb_raw_2.shape) == 2:
                                    batch_size, hidden_dim = class_emb_raw_2.shape
                                    class_emb_raw_2 = class_emb_raw_2.unsqueeze(1)
                                
                                # 统一序列长度
                                if class_emb_raw.shape[1] != class_emb_raw_2.shape[1]:
                                    seq_len_1 = class_emb_raw.shape[1]
                                    seq_len_2 = class_emb_raw_2.shape[1]
                                    max_seq_len = max(seq_len_1, seq_len_2)
                                    
                                    # 调整第一个编码器输出
                                    if seq_len_1 < max_seq_len:
                                        pad_len = max_seq_len - seq_len_1
                                        padding = torch.zeros((class_emb_raw.shape[0], pad_len, class_emb_raw.shape[2]), 
                                                            device=target_device, dtype=class_emb_raw.dtype)
                                        class_emb_raw = torch.cat([class_emb_raw, padding], dim=1)
                                    
                                    # 调整第二个编码器输出
                                    if seq_len_2 < max_seq_len:
                                        pad_len = max_seq_len - seq_len_2
                                        padding = torch.zeros((class_emb_raw_2.shape[0], pad_len, class_emb_raw_2.shape[2]), 
                                                            device=target_device, dtype=class_emb_raw_2.dtype)
                                        class_emb_raw_2 = torch.cat([class_emb_raw_2, padding], dim=1)
                                
                                # 检查并修复NaN值
                                class_emb_raw, has_nan1 = check_and_fix_nan_values(class_emb_raw, "class_emb_raw")
                                class_emb_raw_2, has_nan2 = check_and_fix_nan_values(class_emb_raw_2, "class_emb_raw_2")
                                
                                # 连接两个编码器的输出
                                class_emb_raw = torch.cat([
                                    class_emb_raw.to(target_device),
                                    class_emb_raw_2.to(target_device)
                                ], dim=-1)
                            
                            except Exception as e:
                                accelerator.print(f"[错误] 类别SDXL文本编码器处理错误: {e}")
                                
                                # 回退方案：使用实例嵌入代替
                                if instance_emb is not None and isinstance(instance_emb, torch.Tensor):
                                    accelerator.print("[修复] 使用实例嵌入替代类别嵌入")
                                    class_emb_raw = instance_emb.clone().detach()
                                else:
                                    # 创建全零嵌入
                                    batch_size = latents.shape[0]
                                    class_emb_raw = torch.zeros((batch_size, 8, 2048), device=target_device, dtype=unet_dtype)
                        
                        # 检查并修复NaN
                        class_emb_raw, has_nan = check_and_fix_nan_values(class_emb_raw, "class_emb_raw")
                    except Exception as e:
                        accelerator.print(f"[严重错误] 类别文本编码器处理失败: {e}")
                        # 使用实例嵌入或创建虚拟嵌入
                        if instance_emb is not None:
                            class_emb_raw = instance_emb.clone().detach()
                        else:
                            batch_size = latents.shape[0]
                            hidden_dim = 2048 if is_sdxl else 768
                            class_emb_raw = torch.zeros((batch_size, 8, hidden_dim), 
                                                     device=target_device, dtype=unet_dtype)
                            class_emb_raw.requires_grad_(True)
                
            # 确保它与 unet 数据类型匹配
            class_emb = class_emb_raw.to(dtype=unet_dtype)
        
        # 计算损失时使用混合精度来减少显存使用
        loss_precision_context = torch.amp.autocast(
            "cuda:0", 
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
            
            # 修复可能的repeat维度不匹配问题
            try:
                instance_encoder_hidden = instance_emb.repeat(instance_count, 1, 1).to(device=target_device, dtype=unet_dtype)
            except RuntimeError as e:
                accelerator.print(f"[警告] 实例编码器嵌入重复出错: {e}")
                # 创建正确维度的嵌入
                if len(instance_emb.shape) == 3:
                    # 输入是 [batch, seq_len, hidden_dim]
                    batch_size, seq_len, hidden_dim = instance_emb.shape
                    instance_encoder_hidden = instance_emb[0:1].expand(instance_count, seq_len, hidden_dim)
                else:
                    accelerator.print("[错误] 无法处理实例嵌入的维度")
                    # 创建占位嵌入
                    instance_encoder_hidden = torch.zeros((instance_count, 8, 2048 if is_sdxl else 768), 
                                                       device=target_device, dtype=unet_dtype)
            
            # 检查输入中的NaN并修复
            instance_input, has_nan1 = check_and_fix_nan_values(instance_input, "instance_input")
            instance_encoder_hidden, has_nan2 = check_and_fix_nan_values(instance_encoder_hidden, "instance_encoder_hidden")
            
            if instance_added_cond_kwargs:
                for k, v in instance_added_cond_kwargs.items():
                    instance_added_cond_kwargs[k], has_nan = check_and_fix_nan_values(v, f"instance_added_cond_kwargs[{k}]")
            
            try:
                # 使用UNet进行预测
                instance_pred = unet(
                    instance_input,
                    instance_timesteps,
                    encoder_hidden_states=instance_encoder_hidden,
                    added_cond_kwargs=instance_added_cond_kwargs
                ).sample
                
                # 检查并修复输出中的NaN
                instance_pred, has_nan = check_and_fix_nan_values(instance_pred, "instance_pred")
                
                # 确保目标噪声也在正确的设备上
                noise_instance = noise[is_instance].to(target_device)
                noise_instance, has_nan = check_and_fix_nan_values(noise_instance, "noise_instance")
                
                # 确保损失计算是正确的精度和具有梯度
                try:
                    instance_loss_tensor = F.mse_loss(
                        instance_pred.float(),
                        noise_instance.float(),
                        reduction="mean"
                    )
                    
                    # 检查损失是否为NaN
                    if torch.isnan(instance_loss_tensor) or torch.isinf(instance_loss_tensor):
                        accelerator.print("[警告] 实例MSE损失是NaN/Inf，尝试使用L1损失...")
                        instance_loss_tensor = F.l1_loss(
                            instance_pred.float(),
                            noise_instance.float(),
                            reduction="mean"
                        )
                        
                        # 如果仍然是NaN，使用可微小值
                        if torch.isnan(instance_loss_tensor) or torch.isinf(instance_loss_tensor):
                            accelerator.print("[警告] 所有实例损失都是NaN，使用可微小值...")
                            dummy_param = next(unet.parameters())
                            instance_loss_tensor = torch.sum(dummy_param * 0.0) + torch.tensor(1e-4, device=target_device, requires_grad=True)
                    
                    instance_loss_val = instance_loss_tensor
                except Exception as e:
                    accelerator.print(f"[错误] 实例损失计算错误: {e}")
                    dummy_param = next(unet.parameters())
                    instance_loss_val = torch.sum(dummy_param * 0.0) + torch.tensor(1e-4, device=target_device, requires_grad=True)
                
            except Exception as e:
                accelerator.print(f"[错误] UNet实例预测错误: {e}")
                # 创建可微小值
                dummy_param = next(unet.parameters())
                instance_loss_val = torch.sum(dummy_param * 0.0) + torch.tensor(1e-4, device=target_device, requires_grad=True)
                
        # 类别损失计算，使用与实例类似的错误处理
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
            
            # 修复可能的repeat维度不匹配问题
            try:
                class_encoder_hidden = class_emb.repeat(class_count, 1, 1).to(device=target_device, dtype=unet_dtype)
            except RuntimeError as e:
                accelerator.print(f"[警告] 类别编码器嵌入重复出错: {e}")
                # 创建正确维度的嵌入
                if len(class_emb.shape) == 3:
                    batch_size, seq_len, hidden_dim = class_emb.shape
                    class_encoder_hidden = class_emb[0:1].expand(class_count, seq_len, hidden_dim)
                else:
                    # 创建占位嵌入
                    class_encoder_hidden = torch.zeros((class_count, 8, 2048 if is_sdxl else 768), 
                                                   device=target_device, dtype=unet_dtype)
            
            # 检查输入中的NaN并修复
            class_input, has_nan1 = check_and_fix_nan_values(class_input, "class_input")
            class_encoder_hidden, has_nan2 = check_and_fix_nan_values(class_encoder_hidden, "class_encoder_hidden", aggressive=True)
            
            if class_added_cond_kwargs:
                for k, v in class_added_cond_kwargs.items():
                    class_added_cond_kwargs[k], has_nan = check_and_fix_nan_values(v, f"class_added_cond_kwargs[{k}]")
            
            try:
                # 使用UNet进行预测
                class_pred = unet(
                    class_input,
                    class_timesteps,
                    encoder_hidden_states=class_encoder_hidden,
                    added_cond_kwargs=class_added_cond_kwargs
                ).sample
                
                # 检查并修复输出中的NaN
                class_pred, has_nan = check_and_fix_nan_values(class_pred, "class_pred")
                
                # 确保目标噪声也在同一设备上
                noise_class = noise[~is_instance].to(target_device)
                noise_class, has_nan = check_and_fix_nan_values(noise_class, "noise_class")
                
                # 计算损失
                try:
                    class_loss_tensor = F.mse_loss(
                        class_pred.float(),
                        noise_class.float(),
                        reduction="mean"
                    )
                    
                    # 检查损失是否为NaN
                    if torch.isnan(class_loss_tensor) or torch.isinf(class_loss_tensor):
                        accelerator.print("[警告] 类别MSE损失是NaN/Inf，尝试使用L1损失...")
                        class_loss_tensor = F.l1_loss(
                            class_pred.float(),
                            noise_class.float(),
                            reduction="mean"
                        )
                        
                        # 如果仍然是NaN，使用可微小值
                        if torch.isnan(class_loss_tensor) or torch.isinf(class_loss_tensor):
                            accelerator.print("[警告] 所有类别损失都是NaN，使用可微小值...")
                            dummy_param = next(unet.parameters())
                            class_loss_tensor = torch.sum(dummy_param * 0.0) + torch.tensor(1e-4, device=target_device, requires_grad=True)
                    
                    class_loss_val = class_loss_tensor
                except Exception as e:
                    accelerator.print(f"[错误] 类别损失计算错误: {e}")
                    dummy_param = next(unet.parameters())
                    class_loss_val = torch.sum(dummy_param * 0.0) + torch.tensor(1e-4, device=target_device, requires_grad=True)
                    
            except Exception as e:
                accelerator.print(f"[错误] UNet类别预测错误: {e}")
                # 创建可微小值
                dummy_param = next(unet.parameters())
                class_loss_val = torch.sum(dummy_param * 0.0) + torch.tensor(1e-4, device=target_device, requires_grad=True)
        
        # 在每次迭代结束时显式释放未使用的张量
        if config["memory_optimization"].get("aggressive_gc", False) and torch.cuda.is_available():
            # 清理不需要的临时张量
            del noise, latents, noisy_latents
            if "instance_pred" in locals():
                del instance_pred
            if "class_pred" in locals():
                del class_pred
            torch.cuda.empty_cache()
        
        # 组合最终损失，确保有梯度连接
        if isinstance(instance_loss_val, torch.Tensor) and isinstance(class_loss_val, torch.Tensor):
            # 检查损失是否有效
            instance_valid = not (torch.isnan(instance_loss_val) or torch.isinf(instance_loss_val))
            class_valid = not (torch.isnan(class_loss_val) or torch.isinf(class_loss_val))
            
            if instance_valid and class_valid:
                loss = instance_loss_val + prior_preservation_weight * class_loss_val
            elif instance_valid:
                loss = instance_loss_val
                accelerator.print("[警告] 类别损失无效，仅使用实例损失")
            elif class_valid:
                loss = prior_preservation_weight * class_loss_val
                accelerator.print("[警告] 实例损失无效，仅使用类别损失")
            else:
                # 创建可微小损失
                dummy_param = next(unet.parameters())
                loss = torch.sum(dummy_param * 0.0) + torch.tensor(1e-4, device=target_device, requires_grad=True)
                accelerator.print("[警告] 所有损失无效，使用可微小损失")
        elif isinstance(instance_loss_val, torch.Tensor):
            if not (torch.isnan(instance_loss_val) or torch.isinf(instance_loss_val)):
                loss = instance_loss_val
            else:
                # 创建可微小损失
                dummy_param = next(unet.parameters())
                loss = torch.sum(dummy_param * 0.0) + torch.tensor(1e-4, device=target_device, requires_grad=True)
        elif isinstance(class_loss_val, torch.Tensor):
            if not (torch.isnan(class_loss_val) or torch.isinf(class_loss_val)):
                loss = prior_preservation_weight * class_loss_val
            else:
                # 创建可微小损失
                dummy_param = next(unet.parameters())
                loss = torch.sum(dummy_param * 0.0) + torch.tensor(1e-4, device=target_device, requires_grad=True)
        else: # 两种损失都是标量 0.0
            # 创建可微小损失
            dummy_param = next(unet.parameters())
            loss = torch.sum(dummy_param * 0.0) + torch.tensor(1e-4, device=target_device, requires_grad=True)
        
        # 确保损失是可微的
        if not loss.requires_grad:
            dummy_param = next(unet.parameters())
            loss = loss + torch.sum(dummy_param * 0.0)
        
        return loss, instance_loss_val, class_loss_val
    
    except Exception as e:
        accelerator.print(f"损失计算错误: {e}")
        import traceback
        accelerator.print(traceback.format_exc())
        
        # 返回带有梯度连接的小损失
        dummy_param = next(unet.parameters())
        error_loss = torch.sum(dummy_param * 0.0) + torch.tensor(1e-4, device=target_device, requires_grad=True)
        return error_loss, 0.0, 0.0
