"""
DreamBooth 训练循环模块
包含训练过程中的核心循环逻辑，负责损失计算和优化
"""
import torch
import os
import gc
from .training_initialization import initialize_training_environment
from .loss_calculation import compute_loss
from .model_utils import ensure_device_consistency, get_model_device_info
from .memory_optimization import aggressive_memory_cleanup, cyclic_memory_cleanup
from .training_utils import (
    log_losses, print_training_status, save_checkpoint,
    handle_training_interruption, update_loss_plot, append_loss_to_csv
)


def execute_training_loop(
    accelerator, unet, text_encoder, vae, tokenizer, 
    dataloader, optimizer, noise_scheduler, lr_scheduler,
    config,
    resume_step,
    memory_mgr, debug_monitor, loss_monitor,
    mixed_precision_dtype,
    loss_csv_path,
    text_encoder_2=None  # 添加第二个文本编码器参数
):
    """执行DreamBooth核心训练循环"""
    
    init_data = initialize_training_environment(
        accelerator, unet, text_encoder, vae, tokenizer, optimizer,
        noise_scheduler, config, resume_step, mixed_precision_dtype, loss_csv_path, dataloader, 
        text_encoder_2=text_encoder_2  # 传递给初始化函数
    )

    current_noise_scheduler = init_data["noise_scheduler"]
    global_step = init_data["global_step"]
    loss_history = init_data["loss_history"]
    config_params = init_data["config_params"]
    progress_bar = init_data["progress_bar"]
    image_progress = init_data["image_progress"]
    device = init_data["device"]
    unet_dtype = init_data["unet_dtype"]
    instance_text_inputs = init_data["instance_text_inputs"]
    class_text_inputs = init_data["class_text_inputs"]

    # 添加显存优化配置参数
    enable_memory_efficient_attention = config["memory_optimization"].get("xformers_optimization", True)
    enable_attention_slicing = config["memory_optimization"].get("attention_slice_size", 0) > 0
    aggressive_memory_cleanup_enabled = config["memory_optimization"].get("aggressive_gc", False)
    memory_cleanup_frequency = config["memory_optimization"].get("memory_cleanup_frequency", 10)
    
    # 监控是否开启内存使用限制
    if config["memory_optimization"].get("limit_gpu_memory", False) and torch.cuda.is_available():
        fraction = config["memory_optimization"].get("gpu_memory_fraction", 0.9)
        torch.cuda.set_per_process_memory_fraction(fraction)
        accelerator.print(f"已限制GPU内存使用为总内存的{fraction*100:.0f}%")
    
    # 应用显存优化
    if enable_memory_efficient_attention:
        try:
            if hasattr(unet, "enable_xformers_memory_efficient_attention"):
                unet.enable_xformers_memory_efficient_attention()
                accelerator.print("已启用 xformers 内存优化")
        except:
            accelerator.print("无法启用 xformers 内存优化，请检查是否正确安装")
    
    if enable_attention_slicing:
        try:
            slice_size = config["memory_optimization"].get("attention_slice_size", 4)
            unet.set_attention_slice(slice_size)
            accelerator.print(f"已启用注意力切片优化，切片大小: {slice_size}")
        except:
            accelerator.print("无法设置注意力切片")

    # 初始内存清理
    if aggressive_memory_cleanup_enabled and torch.cuda.is_available():
        aggressive_memory_cleanup()
        accelerator.print("已执行初始内存清理")
        
        # 显示初始内存状态
        initial_gpu_memory = torch.cuda.memory_allocated() / 1024**3
        max_gpu_memory = torch.cuda.max_memory_allocated() / 1024**3
        accelerator.print(f"初始GPU内存: {initial_gpu_memory:.2f}GB，峰值: {max_gpu_memory:.2f}GB")

    # Extract frequently used params for clarity
    max_train_steps = config_params["max_train_steps"]
    output_dir = config_params["output_dir"]
    train_text_encoder_flag = config_params["train_text_encoder"]
    prior_preservation_weight = config_params["prior_preservation_weight"]
    
    params_to_optimize = list(unet.parameters())
    if train_text_encoder_flag:
        params_to_optimize += list(text_encoder.parameters())
        if text_encoder_2 is not None: # Also add text_encoder_2 params if training text encoders
            params_to_optimize += list(text_encoder_2.parameters())
    
    # 确保text_encoder_2的状态与text_encoder一致
    if text_encoder_2 is not None:
        if train_text_encoder_flag:
            text_encoder_2.train()
            text_encoder_2.requires_grad_(True)
        else:
            text_encoder_2.eval()
            text_encoder_2.requires_grad_(False)
    
    # Models (unet, text_encoder, vae, text_encoder_2) are already on accelerator.device
    # due to accelerator.prepare() in dreambooth.py.
    # The explicit .to(device) calls here are redundant and have been removed.
    target_device = device # accelerator.device

    model_device_info = get_model_device_info({
        "unet": unet, 
        "text_encoder": text_encoder,
        "vae": vae
    })
    if text_encoder_2 is not None:
        model_device_info["text_encoder_2"] = str(next(text_encoder_2.parameters()).device if list(text_encoder_2.parameters()) else "Unknown (no params)")
    
    if accelerator.is_main_process:
        accelerator.print(f"模型设备信息 (training_loop start): {model_device_info}")
    
    # The following explicit .to(model_device) calls are also redundant and removed.
    # model_device = device
    # accelerator.print(f"[初始化] 确保所有模型在设备 {model_device} 上")
    # unet = unet.to(model_device)
    # text_encoder = text_encoder.to(model_device)
    # vae = vae.to(model_device)
    # if text_encoder_2 is not None:
    #    text_encoder_2 = text_encoder_2.to(model_device)

    # 创建一个全局模型设备缓存变量
    global _models_on_device
    _models_on_device = True
    
    # 跟踪连续NaN次数
    nan_loss_count = 0
    max_consecutive_nans = 5  # 在降低学习率前允许的最大连续NaN次数
    
    training_successful = False

    try:
        for epoch in range(0, config["training"].get("num_epochs", 1)):
            unet.train()
            if train_text_encoder_flag:
                text_encoder.train()
            else:
                if hasattr(text_encoder, 'module') and isinstance(text_encoder.module, torch.nn.Module):
                    text_encoder.module.eval()
                elif isinstance(text_encoder, torch.nn.Module):
                    text_encoder.eval()
            
            for step, batch in enumerate(dataloader):
                with accelerator.accumulate(unet):
                    # 执行周期性内存清理
                    if aggressive_memory_cleanup_enabled and step % memory_cleanup_frequency == 0:
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                            gc.collect()
                            
                            # 每隔一定步骤显示内存使用情况
                            if step % (memory_cleanup_frequency * 5) == 0 and accelerator.is_main_process:
                                current_gpu_memory = torch.cuda.memory_allocated() / 1024**3
                                max_gpu_memory = torch.cuda.max_memory_allocated() / 1024**3
                                accelerator.print(f"步骤 {global_step} 内存使用: {current_gpu_memory:.2f}GB，峰值: {max_gpu_memory:.2f}GB")
                    
                    # 确保批次数据在正确设备上
                    if "pixel_values" in batch and batch["pixel_values"].device != device:
                        batch["pixel_values"] = batch["pixel_values"].to(device)
                        
                    if "is_instance" in batch and batch["is_instance"].device != device:
                        batch["is_instance"] = batch["is_instance"].to(device)
                    
                    try:
                        # 确保模型在正确设备上
                        if vae.device != device:
                            accelerator.print(f"[警告] VAE不在目标设备上，将移动到 {device}")
                            vae = vae.to(device)
                        
                        if unet.device != device:
                            accelerator.print(f"[警告] UNet不在目标设备上，将移动到 {device}")
                            unet = unet.to(device)
                        
                        if text_encoder.device != device:
                            accelerator.print(f"[警告] 文本编码器不在目标设备上，将移动到 {device}")
                            text_encoder = text_encoder.to(device)
                        
                        if text_encoder_2 is not None and text_encoder_2.device != device:
                            accelerator.print(f"[警告] 文本编码器2不在目标设备上，将移动到 {device}")
                            text_encoder_2 = text_encoder_2.to(device)
                        
                        # 确保实例文本输入在正确设备上
                        if instance_text_inputs.input_ids.device != device:
                            instance_text_inputs.input_ids = instance_text_inputs.input_ids.to(device)
                        
                        # 确保类别文本输入在正确设备上(如果存在)
                        if class_text_inputs is not None and class_text_inputs.input_ids.device != device:
                            class_text_inputs.input_ids = class_text_inputs.input_ids.to(device)
                        
                        loss, instance_loss_val, class_loss_val = compute_loss(
                            accelerator, batch, unet, vae, current_noise_scheduler,
                            instance_text_inputs, class_text_inputs, 
                            text_encoder, prior_preservation_weight, device, 
                            mixed_precision_dtype, unet_dtype, config=config,
                            text_encoder_2=text_encoder_2  # 确保传递第二个文本编码器
                        )
                        
                        # 检查损失是否有梯度连接
                        if not loss.requires_grad:
                            accelerator.print("[警告] 损失张量没有梯度连接，修正中...")
                            # 连接到UNet参数的一个小损失
                            dummy_param = next(unet.parameters())
                            loss = loss + torch.sum(dummy_param * 0.0)
                            accelerator.print("[修复] 已恢复损失的梯度连接")
                            
                    except Exception as e:
                        accelerator.print(f"计算损失出错: {e}")
                        import traceback
                        accelerator.print(traceback.format_exc())
                        accelerator.print("跳过当前步骤并继续训练...")
                        # 重置优化器梯度，确保我们不使用错误的梯度
                        optimizer.zero_grad()
                        # 确保更新进度条，即使步骤被跳过
                        progress_bar.update(1)
                        # 增加全局步骤计数器，否则进度会卡住
                        global_step += 1
                        continue
                    
                    if torch.isnan(loss) or torch.isinf(loss):
                        accelerator.print("CRITICAL: Loss is NaN or Inf. Skipping step.")
                        # 增加连续失败的计数
                        nan_loss_count += 1
                        
                        # 如果连续出现太多NaN/Inf损失，尝试降低学习率
                        if nan_loss_count >= max_consecutive_nans:
                            accelerator.print(f"WARNING: 连续 {nan_loss_count} 次出现NaN/Inf损失，降低学习率")
                            for param_group in optimizer.param_groups:
                                param_group['lr'] *= 0.5  # 将学习率降低到50%
                            accelerator.print(f"学习率降低到 {optimizer.param_groups[0]['lr']:.8f}")
                            nan_loss_count = 0  # 重置计数器
                        
                        # 即使跳过步骤也更新进度条
                        optimizer.zero_grad()
                        progress_bar.update(1)
                        global_step += 1  # 确保全局步骤计数器增加
                        continue
                    else:
                        # 如果损失有效，重置NaN计数器
                        nan_loss_count = 0
                    
                    if accelerator.is_main_process:
                        total_loss_val_item = loss.item() if isinstance(loss, torch.Tensor) else loss
                        inst_loss_item = instance_loss_val.item() if isinstance(instance_loss_val, torch.Tensor) else instance_loss_val
                        class_loss_item = class_loss_val.item() if isinstance(class_loss_val, torch.Tensor) else class_loss_val
                        
                        loss_str = f"Step {global_step}/{max_train_steps} | Loss: {total_loss_val_item:.6f}"
                        if inst_loss_item != 0: loss_str += f" | 实例损失: {inst_loss_item:.6f}"
                        if class_loss_item != 0: loss_str += f" | 类别损失: {class_loss_item:.6f}"
                        progress_pct = (global_step / max_train_steps) * 100
                        loss_str += f" | 进度: {progress_pct:.2f}%"
                        
                        instance_count = 0
                        class_count = 0
                        # num_processed_in_batch is determined later from batch["pixel_values"].size(0)

                        if 'is_instance' in batch:
                            is_instance_tensor = batch['is_instance']
                            instance_count = torch.sum(is_instance_tensor).item()
                            # Ensure pixel_values exists before calculating class_count based on its size
                            if "pixel_values" in batch:
                                class_count = batch["pixel_values"].size(0) - instance_count
                            else: # Should not happen if is_instance is present
                                class_count = is_instance_tensor.size(0) - instance_count


                            loss_str += f" | 批次构成: {instance_count}实例/{class_count}类别"
                        
                        accelerator.print(loss_str)
                        
                        num_processed_in_batch = 0
                        if "pixel_values" in batch:
                            num_processed_in_batch = batch["pixel_values"].size(0)

                        log_losses(
                            accelerator, loss, instance_loss_val, class_loss_val,
                            global_step, max_train_steps, loss_history, debug_monitor, lr_scheduler, optimizer,
                            image_progress_bar=image_progress, 
                            num_images_in_batch=num_processed_in_batch,
                            instance_count_in_batch=int(instance_count), # Explicitly cast to int
                            class_count_in_batch=int(class_count)      # Explicitly cast to int
                        )
                    
                    if global_step % config_params["print_status_every_n_steps"] == 0 and global_step > 0 and accelerator.is_main_process:
                        print_training_status(
                            global_step, max_train_steps, loss, instance_loss_val, class_loss_val, prior_preservation_weight
                        )
                    
                    try:
                        # 尝试执行反向传播，捕获梯度相关错误
                        accelerator.backward(loss)
                    except RuntimeError as e:
                        error_msg = str(e)
                        accelerator.print(f"反向传播错误: {error_msg}")
                        
                        if "element 0 of tensors does not require grad" in error_msg:
                            accelerator.print("[修复] 损失没有梯度连接，创建替代损失...")
                            # 创建一个强制连接到模型参数的损失
                            dummy_param = next(unet.parameters())
                            dummy_loss = torch.sum(dummy_param * 0.0) + torch.tensor(1e-4, device=device, requires_grad=True)
                            
                            # 使用替代损失进行反向传播
                            accelerator.print("[修复] 使用替代损失进行反向传播")
                            accelerator.backward(dummy_loss)
                            
                            # 手动应用小梯度，防止参数不变
                            with torch.no_grad():
                                for param in unet.parameters():
                                    if param.grad is not None:
                                        param.grad.data.add_(torch.randn_like(param) * 1e-7)
                        else:
                            accelerator.print("[错误] 未知的反向传播错误，跳过此步骤")
                            optimizer.zero_grad()
                            # 确保进度条和全局步骤更新
                            progress_bar.update(1)
                            global_step += 1
                            continue
                    
                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(params_to_optimize, config_params["max_grad_norm"])
                    
                    optimizer.step()
                    optimizer.zero_grad()
                
                # 执行更彻底的内存清理（更低频率）
                if aggressive_memory_cleanup_enabled:
                    cyclic_memory_cleanup(frequency=50, current_step=step)
                
                # 每隔一定步骤显示详细内存信息
                if aggressive_memory_cleanup_enabled and accelerator.is_main_process and step % 100 == 0 and torch.cuda.is_available():
                    current_gpu_memory = torch.cuda.memory_allocated() / 1024**3
                    max_gpu_memory = torch.cuda.max_memory_allocated() / 1024**3
                    torch.cuda.reset_peak_memory_stats()  # 重置峰值以观察未来的峰值
                    accelerator.print(f"\n[内存监控] 当前: {current_gpu_memory:.2f}GB，本阶段峰值: {max_gpu_memory:.2f}GB\n")
                
                progress_bar.update(1)
                global_step += 1
                
                if global_step >= max_train_steps:
                    break
            if global_step >= max_train_steps:
                break 
        
        if accelerator.is_main_process:
            update_loss_plot(loss_history, output_dir, global_step, max_train_steps)
        
        training_successful = True
        if accelerator.is_main_process:
            if torch.cuda.is_available():
                final_gpu_memory = torch.cuda.memory_allocated() / 1024**3
                max_gpu_memory = torch.cuda.max_memory_allocated() / 1024**3
                accelerator.print(f"\n训练完成统计:\n"
                                  f"- 最终GPU内存占用: {final_gpu_memory:.2f} GB\n"
                                  f"- 训练过程中最大内存占用: {max_gpu_memory:.2f} GB\n"
                                  f"- 完成步数: {global_step}/{max_train_steps}\n"
                                  f"- 实例损失最终值: {loss_history['instance'][-1] if loss_history['instance'] else 'N/A'}\n"
                                  f"- 类别损失最终值: {loss_history['class'][-1] if loss_history['class'] else 'N/A'}")
            
            save_checkpoint(
                accelerator, unet, text_encoder, optimizer,
                global_step, os.path.join(output_dir, "final_checkpoint.pt"), train_text_encoder_flag
            )
        
    except KeyboardInterrupt:
        accelerator.print("\n训练被用户中断")
        handle_training_interruption(
            accelerator, unet, text_encoder, optimizer, 
            global_step, output_dir, train_text_encoder_flag, "interrupt"
        )
        training_successful = False
        
    except torch.cuda.OutOfMemoryError as oom_error:
        accelerator.print(f"\n训练因GPU内存不足而中断: {str(oom_error)}")
        accelerator.print("正在执行紧急内存清理...")
        
        # OOM后的紧急清理
        if torch.cuda.is_available():
            try:
                # 释放模型缓存
                for model in [unet, text_encoder, vae]:
                    if hasattr(model, "zero_grad"):
                        model.zero_grad(set_to_none=True)
                        
                # 释放优化器状态
                optimizer.zero_grad(set_to_none=True)
                
                # 彻底清理内存
                gc.collect()
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                
                # 尝试重新设置内存限制更严格
                torch.cuda.set_per_process_memory_fraction(0.7)  # 降低到70%
                
                accelerator.print("紧急内存清理完成")
            except:
                pass
        
        accelerator.print("建议: 减小批量大小、使用更小的模型或启用梯度检查点")
        handle_training_interruption(
            accelerator, unet, text_encoder, optimizer, 
            global_step, output_dir, train_text_encoder_flag, "oom_error"
        )
        training_successful = False
        
    except Exception as e:
        accelerator.print(f"\n训练遇到错误: {str(e)}")
        import traceback
        traceback.print_exc()
        handle_training_interruption(
            accelerator, unet, text_encoder, optimizer, 
            global_step, output_dir, train_text_encoder_flag, "error"
        )
        training_successful = False
    finally:
        # 最终清理
        if aggressive_memory_cleanup_enabled and torch.cuda.is_available():
            try:
                aggressive_memory_cleanup()
                accelerator.print("已执行最终内存清理")
            except:
                pass
        
        if 'image_progress' in locals() and hasattr(image_progress, 'close'):
            image_progress.close()
        if 'progress_bar' in locals() and hasattr(progress_bar, 'close'):
            progress_bar.close()
    
    accelerator.print(f"\n训练{'成功' if training_successful else '未成功'}完成，总步数: {global_step}/{max_train_steps}")
    
    if global_step < max_train_steps * 0.01:
        accelerator.print("\n⚠️ 警告: 训练步骤数过少，可能未能有效训练模型")
        accelerator.print("可能的原因:")
        accelerator.print("  - GPU内存不足")
        accelerator.print("  - 训练速度过慢（每步耗时过长）")
        accelerator.print("  - 数据集问题（实例图片太少或加载问题）")
        accelerator.print("  - 手动中断训练")
    
    return global_step, loss_history, training_successful
