"""
DreamBooth 模型保存模块
负责保存训练结果和模型可视化
"""
import os
import torch
import matplotlib.pyplot as plt
from diffusers import StableDiffusionPipeline


def save_trained_model(
    pretrained_model_name_or_path, output_dir, unet, text_encoder, 
    tokenizer, noise_scheduler, vae, identifier, 
    loss_history=None, theory_notes_enabled=False
):
    """保存训练完成的模型和辅助信息"""
    # 保存标识符
    os.makedirs(output_dir, exist_ok=True)
    identifier_path = os.path.join(output_dir, "identifier.txt")
    with open(identifier_path, "w") as f:
        f.write(identifier)
    
    # 尝试保存损失图表
    if loss_history and len(loss_history.get("steps", [])) > 0:
        try:
            save_loss_plot(loss_history, output_dir)
        except Exception as e:
            print(f"保存损失图表失败: {e}")
    
    # 保存完整模型
    print("正在保存微调后的模型...")
    pipeline = StableDiffusionPipeline.from_pretrained(
        pretrained_model_name_or_path,
        unet=unet, text_encoder=text_encoder,
        tokenizer=tokenizer, scheduler=noise_scheduler, vae=vae
    )
    pipeline.save_pretrained(output_dir)
    print(f"模型已成功保存到 {output_dir}")
    
    return pipeline


def save_loss_plot(loss_history, output_dir):
    """保存训练损失图表"""
    plt.figure(figsize=(10, 5))
    plt.plot(loss_history["steps"], loss_history["instance"], label="实例损失")
    plt.plot(loss_history["steps"], loss_history["class"], label="类别损失")
    plt.plot(loss_history["steps"], loss_history["total"], label="总损失")
    plt.xlabel("训练步数")
    plt.ylabel("损失值")
    plt.title("DreamBooth 训练损失曲线")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plot_path = os.path.join(output_dir, "training_loss.png")
    plt.savefig(plot_path)
    print(f"训练损失图表已保存至 {plot_path}")
    plt.close()


def load_checkpoint(checkpoint_path, device="cpu"):
    """加载模型检查点"""
    if not os.path.exists(checkpoint_path):
        return None
    
    try:
        return torch.load(checkpoint_path, map_location=device)
    except Exception as e:
        print(f"加载检查点失败: {e}")
        return None


def generate_test_samples(pipeline, identifier, class_name, output_dir, num_samples=3):
    """生成测试样本以验证模型效果"""
    test_prompts = [
        f"a {identifier} {class_name} on the beach",
        f"a {identifier} {class_name} in a forest",
        f"a painting of {identifier} {class_name} in van gogh style"
    ]
    
    test_dir = os.path.join(output_dir, "test_samples")
    os.makedirs(test_dir, exist_ok=True)
    
    results = []
    for i, prompt in enumerate(test_prompts[:num_samples]):
        output = pipeline(prompt, num_inference_steps=50)
        image = output.images[0]
        save_path = os.path.join(test_dir, f"test_sample_{i}.png")
        image.save(save_path)
        results.append((prompt, save_path))
        print(f"测试样本已保存: {save_path}")
    
    return results
