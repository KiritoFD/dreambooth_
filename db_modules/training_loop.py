"""
DreamBooth 训练循环模块
包含训练过程中的核心循环逻辑，负责损失计算和优化
"""
import torch
import os

from .training_initialization import initialize_training_environment
from .loss_calculation import compute_loss
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
    loss_csv_path
):
    """执行DreamBooth核心训练循环"""
    
    init_data = initialize_training_environment(
        accelerator, unet, text_encoder, vae, tokenizer, optimizer,
        noise_scheduler, config, resume_step, mixed_precision_dtype, loss_csv_path, dataloader
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

    # Extract frequently used params for clarity
    max_train_steps = config_params["max_train_steps"]
    output_dir = config_params["output_dir"]
    train_text_encoder_flag = config_params["train_text_encoder"]
    prior_preservation_weight = config_params["prior_preservation_weight"]
    
    params_to_optimize = list(unet.parameters())
    if train_text_encoder_flag:
        params_to_optimize += list(text_encoder.parameters())
    
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
                    loss, instance_loss_val, class_loss_val = compute_loss(
                        accelerator, batch, unet, vae, current_noise_scheduler,
                        instance_text_inputs, class_text_inputs, 
                        text_encoder, prior_preservation_weight, device, 
                        mixed_precision_dtype, unet_dtype, config
                    )
                    
                    if torch.isnan(loss) or torch.isinf(loss):
                        accelerator.print("CRITICAL: Loss is NaN or Inf. Skipping step.")
                        continue
                    
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
                    
                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(params_to_optimize, config_params["max_grad_norm"])
                    
                    optimizer.step()
                    optimizer.zero_grad()
                
                if accelerator.is_main_process:
                    append_loss_to_csv(
                        loss_csv_path, global_step,
                        loss.item() if isinstance(loss, torch.Tensor) else loss,
                        instance_loss_val.item() if isinstance(instance_loss_val, torch.Tensor) else instance_loss_val,
                        class_loss_val.item() if isinstance(class_loss_val, torch.Tensor) else class_loss_val
                    )
                
                if accelerator.is_main_process and global_step % 50 == 0 and global_step > 0: # Plotting frequency
                    update_loss_plot(loss_history, output_dir, global_step, max_train_steps)
                
                if accelerator.is_main_process:
                    if global_step % config_params["save_steps"] == 0 and global_step > 0:
                        save_checkpoint(
                            accelerator, unet, text_encoder, optimizer,
                            global_step, os.path.join(output_dir, "checkpoint.pt"), train_text_encoder_flag
                        )
                
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
