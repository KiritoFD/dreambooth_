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
    
    # 确保text_encoder_2的状态与text_encoder一致
    if text_encoder_2 is not None:
        if train_text_encoder_flag:
            text_encoder_2.train()
            # 如果训练文本编码器，也应该将其添加到优化器参数中
            if train_text_encoder_flag and accelerator.is_main_process:
                accelerator.print("将text_encoder_2添加到优化器参数中")
            params_to_optimize += list(text_encoder_2.parameters())
        else:
            if hasattr(text_encoder_2, 'module') and isinstance(text_encoder_2.module, torch.nn.Module):
                text_encoder_2.module.eval()
            elif isinstance(text_encoder_2, torch.nn.Module):
                text_encoder_2.eval()
    
    # 确保所有模型在同一个设备上
    target_device = device
    
    # 明确将模型移动到目标设备，不依赖accelerator
    if vae.device != target_device:
        print(f"[INFO] 将VAE从 {vae.device} 移动到 {target_device}")
        vae = vae.to(target_device)
    
    if text_encoder.device != target_device:
        print(f"[INFO] 将文本编码器从 {text_encoder.device} 移动到 {target_device}")
        text_encoder = text_encoder.to(target_device)
    
    if unet.device != target_device:
        print(f"[INFO] 将UNet从 {unet.device} 移动到 {target_device}")
        unet = unet.to(target_device)
    
    if text_encoder_2 is not None and text_encoder_2.device != target_device:
        print(f"[INFO] 将文本编码器2从 {text_encoder_2.device} 移动到 {target_device}")
        text_encoder_2 = text_encoder_2.to(target_device)
    
    # 更新dataset的device设置，确保返回的张量已经在正确设备上
    if isinstance(dataloader.dataset, torch.utils.data.Dataset):
        if hasattr(dataloader.dataset, 'device'):
            dataloader.dataset.device = target_device
            print(f"[INFO] 设置数据集设备为 {target_device}")
    
    model_device_info = get_model_device_info({
        "unet": unet, 
        "text_encoder": text_encoder,
        "vae": vae
    })
    if text_encoder_2 is not None:
        model_device_info["text_encoder_2"] = next(text_encoder_2.parameters()).device
    
    if accelerator.is_main_process:
        accelerator.print("[INFO] 模型设备信息:")
        for model_name, device_str in model_device_info.items():
            accelerator.print(f"  - {model_name}: {device_str}")
    
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
                        
                        # 在损失计算后立即执行轻量级内存清理
                        if aggressive_memory_cleanup_enabled and torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    except Exception as e:
                        accelerator.print(f"计算损失出错: {e}")
                        accelerator.print(f"使用torch.cuda.empty_cache()清理显存...")
                        torch.cuda.empty_cache()
                        accelerator.print(f"检查模型设备是否一致...")
                        
                        # 尝试恢复设备一致性
                        model_devices = {
                            "VAE": vae.device,
                            "UNet": unet.device,
                            "文本编码器": text_encoder.device
                        }
                        if text_encoder_2 is not None:
                            model_devices["文本编码器2"] = text_encoder_2.device
                        
                        accelerator.print(f"当前模型设备: {model_devices}")
                        
                        # 尝试将所有模型移到第一个CUDA设备
                        if torch.cuda.is_available():
                            target_cuda = torch.device("cuda:0")
                            accelerator.print(f"尝试将所有模型统一移动到 {target_cuda}")
                            
                            vae = vae.to(target_cuda)
                            unet = unet.to(target_cuda)
                            text_encoder = text_encoder.to(target_cuda)
                            if text_encoder_2 is not None:
                                text_encoder_2 = text_encoder_2.to(target_cuda)
                            
                            # 也移动输入
                            if "pixel_values" in batch:
                                batch["pixel_values"] = batch["pixel_values"].to(target_cuda)
                            if "is_instance" in batch:
                                batch["is_instance"] = batch["is_instance"].to(target_cuda)
                        
                        accelerator.print("跳过当前步骤并继续训练...")
                        optimizer.zero_grad()
                        continue
                    
                    if torch.isnan(loss) or torch.isinf(loss):
                        accelerator.print("CRITICAL: Loss is NaN or Inf. Skipping step.")
                        # 增加连续失败的计数
                        if "nan_loss_count" not in locals():
                            nan_loss_count = 0
                        nan_loss_count += 1
                        
                        # 如果连续出现太多NaN/Inf损失，尝试降低学习率
                        if nan_loss_count >= 5:
                            accelerator.print(f"WARNING: 连续 {nan_loss_count} 次出现NaN/Inf损失，尝试降低学习率")
                            for param_group in optimizer.param_groups:
                                param_group['lr'] *= 0.8  # 将学习率降低到80%
                            accelerator.print(f"学习率降低到 {optimizer.param_groups[0]['lr']:.8f}")
                            nan_loss_count = 0  # 重置计数器
                            
                        optimizer.zero_grad()
                        continue
                    else:
                        # 如果损失有效，重置NaN计数器
                        if "nan_loss_count" in locals():
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
                            instance_count_in_batch=instance_count, # Pass determined counts
                            class_count_in_batch=class_count      # Pass determined counts
                        )
                    
                    if global_step % config_params["print_status_every_n_steps"] == 0 and global_step > 0 and accelerator.is_main_process:
                        print_training_status(
                            global_step, max_train_steps, loss, instance_loss_val, class_loss_val, prior_preservation_weight
                        )
                    
                    accelerator.backward(loss)
                    
                    # 在反向传播后进行清理，这是内存使用的高峰点
                    if aggressive_memory_cleanup_enabled and torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    
                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(params_to_optimize, config_params["max_grad_norm"])
                    
                    optimizer.step()
                    optimizer.zero_grad()
                    
                    # 优化器步骤后清理
                    if aggressive_memory_cleanup_enabled and torch.cuda.is_available() and step % 5 == 0:
                        gc.collect()
                        torch.cuda.empty_cache()
                
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
