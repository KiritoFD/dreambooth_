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
    dataloader, optimizer, noise_scheduler, instance_prompt, class_prompt,
    max_train_steps, output_dir, prior_preservation_weight=1.0, 
    gradient_accumulation_steps=1, train_text_encoder=True,
    resume_from=None, memory_mgr=None, debug_monitor=None,
    has_theory_notes=False, update_stage_fn=None, get_theory_step=None,
    max_grad_norm=1.0,  # 添加梯度裁剪参数
    low_memory=False    # 添加低内存模式标志
):
    """执行DreamBooth核心训练循环"""
    # GPU内存监控
    if torch.cuda.is_available():
        starting_gpu_memory = torch.cuda.memory_allocated() / 1024**3
        print(f"开始训练循环前GPU内存占用: {starting_gpu_memory:.2f} GB")
    
    # 进度条和优化参数设置
    if resume_from and resume_from > 0:
        global_step = resume_from
        progress_bar = tqdm(range(global_step, max_train_steps), 
                           initial=global_step, total=max_train_steps,
                           desc="训练进度")
    else:
        global_step = 0
        progress_bar = tqdm(range(max_train_steps), desc="训练进度")
        
    # 训练过程监控变量
    early_stop_threshold = 3
    no_improvement_count = 0
    last_loss = float('inf')
    save_checkpoint_steps = 50
    checkpoint_path = os.path.join(output_dir, "checkpoint.pt")
    
    # 记录损失值
    loss_history = {"instance": [], "class": [], "total": [], "steps": []}
    
    # 训练开始阶段标记
    if has_theory_notes and update_stage_fn:
        update_stage_fn()  # 第7阶段：训练循环开始
        step_info = get_theory_step("training_loop") if get_theory_step else None
        if step_info:
            print("\n" + "-"*60)
            print("训练循环开始")
            print("-"*60)
            print(step_info["description"])

    # 准备文本输入并移至设备上
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
    ).to(device)
    
    # 参数准备
    params_to_optimize = list(unet.parameters())
    if train_text_encoder:
        params_to_optimize += list(text_encoder.parameters())
    
    # 检查训练前的模型状态
    print(f"训练前UNet参数总量: {sum(p.numel() for p in unet.parameters())/1e6:.2f}M")
    
    # 启用低内存模式优化
    if low_memory:
        print("启用低内存训练模式")
        # 主动清理不必要的缓存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # 主要训练循环
    try:
        for epoch in range(1):  # 通常一个epoch就足够
            unet.train()
            text_encoder.train() if train_text_encoder else text_encoder.eval()
            
            # 为了调试目的，预先确定数据集大小
            dataset_size = len(dataloader.dataset)
            print(f"开始训练，数据集包含 {dataset_size} 个样本")
            
            # 迭代数据批次
            for step, batch in enumerate(dataloader):
                # 在第一个批次检查数据形状
                if step == 0 and global_step == 0:
                    print(f"批次大小: {batch['pixel_values'].shape}")
                    print(f"实例样本数: {torch.sum(batch['is_instance']).item()}")
                    print(f"类别样本数: {torch.sum(~batch['is_instance']).item()}")
                
                # 内存清理
                if memory_mgr and step % 5 == 0:
                    memory_mgr.cleanup()
                
                # 中期检查点
                if global_step == max_train_steps // 2 and global_step > 0:
                    # 训练循环中期
                    if has_theory_notes and update_stage_fn:
                        update_stage_fn()  # 第8阶段：训练循环中期
                        print("\n" + "-"*60)
                        print(f"训练已完成50%: {global_step}/{max_train_steps}步")
                        print("-"*60)

                # 梯度累积
                with accelerator.accumulate(unet):
                    try:
                        # 主要训练步骤
                        loss, instance_loss, class_loss = compute_loss(
                            accelerator, batch, unet, vae, noise_scheduler,
                            instance_text_inputs, class_text_inputs, text_encoder, 
                            prior_preservation_weight, device
                        )
                        
                        # 检查损失是否是合法值
                        if torch.isnan(loss) or torch.isinf(loss):
                            print(f"警告: 步骤 {global_step} 出现非法损失值: {loss.item()}")
                            continue
                        
                        # 记录损失
                        if accelerator.is_main_process and global_step % 10 == 0:
                            log_losses(
                                accelerator, loss, instance_loss, class_loss,
                                global_step, max_train_steps, loss_history, debug_monitor
                            )
                        
                        # 每200步打印详细状态
                        if global_step % 200 == 0 and global_step > 0 and has_theory_notes:
                            print_training_status(
                                global_step, max_train_steps, loss, instance_loss, class_loss, prior_preservation_weight
                            )
                        
                        # 损失改善检测
                        current_loss = loss.detach().item()
                        if abs(current_loss - last_loss) < 1e-5:
                            no_improvement_count += 1
                        else:
                            no_improvement_count = 0
                        last_loss = current_loss
                        
                        # 反向传播
                        accelerator.backward(loss)
                        
                        # 梯度裁剪
                        if accelerator.sync_gradients:
                            accelerator.clip_grad_norm_(params_to_optimize, max_grad_norm)
                        
                        # 检查梯度是否存在
                        if global_step == 0 and step == 0:
                            has_grad = any(p.grad is not None and torch.sum(torch.abs(p.grad)) > 0 
                                        for p in params_to_optimize)
                            print(f"梯度检查: {'存在有效梯度' if has_grad else '梯度全为零或不存在'}")
                        
                        # 优化步骤
                        optimizer.step()
                        optimizer.zero_grad()
                    except Exception as step_error:
                        print(f"训练步骤 {global_step} 出错: {step_error}")
                        if global_step == 0:  # 如果是第一步就出错，可能是内存问题
                            raise  # 重新抛出异常，终止训练
                        continue  # 否则尝试继续下一步
                
                # 日志记录和检查点保存
                if global_step % 25 == 0 and accelerator.is_main_process:  # 频率改为25步
                    print(f"\n步骤 {global_step}/{max_train_steps}: 总损失={loss.detach().item():.4f}, "
                        f"实例={instance_loss.detach().item() if isinstance(instance_loss, torch.Tensor) else instance_loss:.4f}, "
                        f"类别={class_loss.detach().item() if isinstance(class_loss, torch.Tensor) else class_loss:.4f}")
                    
                    # 保存检查点
                    if accelerator.is_local_main_process:
                        save_checkpoint(
                            accelerator, unet, text_encoder, optimizer,
                            global_step, checkpoint_path, train_text_encoder
                        )
                        
                        # 报告GPU内存使用情况
                        if torch.cuda.is_available():
                            current_memory = torch.cuda.memory_allocated() / 1024**3
                            max_memory = torch.cuda.max_memory_allocated() / 1024**3
                            print(f"GPU内存: 当前 {current_memory:.2f}GB, 峰值 {max_memory:.2f}GB")
                
                # 更新进度条和步骤计数
                progress_bar.update(1)
                global_step += 1
                
                # 检查是否完成训练
                if global_step >= max_train_steps:
                    break
                
                # 早停检查
                if no_improvement_count >= early_stop_threshold:
                    print(f"\n警告: 连续 {early_stop_threshold} 步损失没有明显改善")
                    print("训练将继续进行，这可能是优化过程的正常波动")
                    no_improvement_count = 0  # 重置计数器
        
        # 训练完成标记
        if has_theory_notes and update_stage_fn:
            update_stage_fn()  # 第9阶段：训练循环结束
            print("\n" + "-"*60)
            print("训练循环完成")
            print("-"*60)
        
        if debug_monitor:
            debug_monitor.log_completion(global_step, max_train_steps, success=True)
            
        training_successful = True
        
    except KeyboardInterrupt:
        print("\n训练被用户中断")
        if debug_monitor:
            debug_monitor.log_error("用户手动中断训练", global_step, max_train_steps)
        
        # 保存中断检查点
        handle_training_interruption(
            accelerator, unet, text_encoder, optimizer, 
            global_step, output_dir, train_text_encoder, "interrupt"
        )
        training_successful = False
        
    except Exception as e:
        print(f"\n训练遇到错误: {str(e)}")
        if debug_monitor:
            debug_monitor.log_error(e, global_step, max_train_steps)
        
        # 记录异常堆栈并保存错误检查点
        import traceback
        traceback.print_exc()
        
        handle_training_interruption(
            accelerator, unet, text_encoder, optimizer, 
            global_step, output_dir, train_text_encoder, "error"
        )
        training_successful = False
    
    # 返回训练结果
    return global_step, loss_history, training_successful


def compute_loss(
    accelerator, batch, unet, vae, noise_scheduler,
    instance_text_inputs, class_text_inputs, text_encoder, 
    prior_preservation_weight, device
):
    """计算DreamBooth训练损失"""
    try:
        # 准备输入
        pixel_values = batch["pixel_values"].to(device)
        is_instance = batch["is_instance"]
        
        # 确保输入数据有效
        if pixel_values.shape[0] == 0:
            return torch.tensor(0.0), 0.0, 0.0
        
        # 编码到潜在空间
        with torch.no_grad():
            latents = vae.encode(pixel_values.to(dtype=vae.dtype)).latent_dist.sample()
            latents = latents * vae.config.scaling_factor
        
        # 添加噪声
        noise = torch.randn_like(latents)
        timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, 
                                (latents.shape[0],), device=latents.device).long()
        noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
        
        # 准备文本嵌入
        with torch.no_grad():
            instance_emb = None
            class_emb = None
            
            # 只在有实例样本时计算实例嵌入
            if torch.sum(is_instance).item() > 0:
                instance_emb = text_encoder(
                    instance_text_inputs.input_ids
                )[0]
            
            # 只在有类别样本时计算类别嵌入
            if torch.sum(~is_instance).item() > 0:
                class_emb = text_encoder(
                    class_text_inputs.input_ids
                )[0]

        # 计算损失
        instance_loss = 0.0
        class_loss = 0.0
        
        # 实例损失(特定主体)
        instance_count = torch.sum(is_instance).item()
        if instance_count > 0:
            instance_pred = unet(
                noisy_latents[is_instance],
                timesteps[is_instance],
                encoder_hidden_states=instance_emb.repeat(instance_count, 1, 1)
            ).sample
            
            instance_loss = F.mse_loss(
                instance_pred.float(),
                noise[is_instance].float(),
                reduction="mean"
            )
        
        # 类别损失(先验保留)
        class_count = torch.sum(~is_instance).item()
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
        
        # 组合损失 (论文公式1)
        if isinstance(instance_loss, torch.Tensor) and isinstance(class_loss, torch.Tensor):
            loss = instance_loss + prior_preservation_weight * class_loss
        elif isinstance(instance_loss, torch.Tensor):
            loss = instance_loss
        elif isinstance(class_loss, torch.Tensor):
            loss = prior_preservation_weight * class_loss
        else:
            # 如果两种损失都是0，返回零张量
            loss = torch.tensor(0.0, device=device)
        
        return loss, instance_loss, class_loss
    
    except Exception as e:
        print(f"损失计算错误: {e}")
        # 回退到零损失，但记录错误
        import traceback
        traceback.print_exc()
        zero_tensor = torch.tensor(0.0, device=device)
        return zero_tensor, 0.0, 0.0


def log_losses(accelerator, loss, instance_loss, class_loss, global_step, max_train_steps, loss_history, debug_monitor=None):
    """记录训练损失"""
    il = instance_loss.detach().item() if isinstance(instance_loss, torch.Tensor) else instance_loss
    cl = class_loss.detach().item() if isinstance(class_loss, torch.Tensor) else class_loss
    tl = loss.detach().item()
    
    loss_history["instance"].append(il)
    loss_history["class"].append(cl)
    loss_history["total"].append(tl)
    loss_history["steps"].append(global_step)
    
    # 记录调试信息
    if debug_monitor:
        debug_monitor.log_step(global_step, max_train_steps, {
            "instance_loss": il,
            "class_loss": cl,
            "total_loss": tl
        })


def print_training_status(global_step, max_train_steps, loss, instance_loss, class_loss, prior_preservation_weight):
    """打印详细训练状态"""
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
- 实例损失: {instance_loss.detach().item() if isinstance(instance_loss, torch.Tensor) else instance_loss:.6f}
- 类别损失: {class_loss.detach().item() if isinstance(class_loss, torch.Tensor) else class_loss:.6f}
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
    print(f"保存检查点到步骤 {global_step}")


def handle_training_interruption(accelerator, unet, text_encoder, optimizer, global_step, output_dir, train_text_encoder, reason="interrupt"):
    """处理训练中断情况"""
    print("\n尝试保存当前模型状态...")
    
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
            print(f"{reason.capitalize()}检查点已保存，您可以稍后恢复训练")
            return True
        except:
            print(f"保存{reason}检查点失败")
    
    return False
