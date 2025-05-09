import torch
import os
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import csv

def log_losses(accelerator, loss, instance_loss, class_loss, global_step, max_train_steps, loss_history, 
               debug_monitor=None, lr_scheduler=None, optimizer=None, 
               image_progress_bar=None, num_images_in_batch=0,
               instance_count_in_batch=0, class_count_in_batch=0): # Added instance/class counts
    """记录训练损失并更新图像进度条及其描述"""
    il = instance_loss.item() if isinstance(instance_loss, torch.Tensor) else instance_loss
    cl = class_loss.item() if isinstance(class_loss, torch.Tensor) else class_loss
    tl = loss.detach().item()
    
    loss_history["instance"].append(float(il))
    loss_history["class"].append(float(cl))
    loss_history["total"].append(float(tl))
    loss_history["steps"].append(global_step)
    
    # 添加训练进度打印
    # progress = global_step / max_train_steps * 100 # This variable is not used further
    
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

    if image_progress_bar and accelerator.is_local_main_process:
        if num_images_in_batch > 0:
            image_progress_bar.update(num_images_in_batch)

        postfix_parts = []
        # image_progress_bar.n is the current count, image_progress_bar.total is the total
        if hasattr(image_progress_bar, 'total') and image_progress_bar.total is not None and image_progress_bar.total > 0 :
            postfix_parts.append(f"{image_progress_bar.n}/{image_progress_bar.total} imgs")

        if instance_count_in_batch > 0 or class_count_in_batch > 0:
            postfix_parts.append(f"Cur Batch: {instance_count_in_batch}i/{class_count_in_batch}c")
        elif num_images_in_batch > 0: # If batch composition not available, but images were processed
            postfix_parts.append(f"Cur Batch: {num_images_in_batch} imgs")
        
        if postfix_parts:
            image_progress_bar.set_postfix_str(", ".join(postfix_parts), refresh=True)
        elif num_images_in_batch > 0: # If only num_images_in_batch is available
             image_progress_bar.set_postfix_str(f"Processed: {image_progress_bar.n}", refresh=True)
        # If no relevant info, tqdm will show its default. No need for "Waiting..." unless desired.

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

def update_loss_plot(loss_history, output_dir, global_step, max_train_steps):
    """实时更新损失曲线并保存"""
    try:
        # import matplotlib.pyplot as plt # Already imported at the top
        # from matplotlib.ticker import MaxNLocator # Already imported at the top
        
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
            os.fsync(file.fileno()) # fsync might not be available on all OS, or file might not have fileno if not a real file.
                                    # Consider if this strictness is always needed.
    except Exception as e:
        print(f"写入损失值到CSV时出错: {e}")
