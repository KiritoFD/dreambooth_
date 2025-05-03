"""
DreamBooth 先验图像生成模块
负责生成用于先验保留损失的类别图像
"""
import os
import torch
from tqdm.auto import tqdm


def generate_prior_images(
    pipeline, class_prompt, output_dir, num_samples=200,
    batch_size=None, theory_notes_enabled=False, theory_step_fn=None
):
    """生成类别先验图像"""
    # 打印先验保留理论
    if theory_notes_enabled and theory_step_fn:
        theory = theory_step_fn("prior_preservation")
        if theory:
            from theory_notes import print_theory_step
            print_theory_step("2", theory["title"], theory["description"])
    
    # 创建输出目录
    class_images_dir = os.path.join(output_dir, "class_images")
    os.makedirs(class_images_dir, exist_ok=True)
    
    # 检查现有图像
    existing_images = len([f for f in os.listdir(class_images_dir) 
                         if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    if existing_images >= num_samples:
        print(f"\n在'{class_images_dir}'目录中找到{existing_images}张现有类别图像，跳过生成步骤")
        return class_images_dir
    
    print(f"\n开始生成{num_samples}张类别图像用于先验保留...")
    
    # 禁用安全检查器以加速生成
    pipeline.safety_checker = None
    
    # 自动确定批次大小
    if batch_size is None:
        if torch.cuda.is_available():
            total_vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
            if total_vram < 6:
                batch_size = 1
            elif total_vram < 12:
                batch_size = 2
            else:
                batch_size = 4
        else:
            batch_size = 1
    
    num_batches = (num_samples + batch_size - 1) // batch_size
    
    # 批量生成图像
    for batch_idx in tqdm(range(num_batches), desc="生成类别图像"):
        batch_prompts = [class_prompt] * min(batch_size, num_samples - batch_idx * batch_size)
        with torch.no_grad():
            outputs = pipeline(batch_prompts, num_inference_steps=50, guidance_scale=7.5)
            
        for i, image in enumerate(outputs.images):
            img_idx = batch_idx * batch_size + i
            image.save(os.path.join(class_images_dir, f"class_{img_idx:04d}.png"))
    
    print(f"成功生成了 {num_samples} 张类别图像，已保存到 {class_images_dir}")
    return class_images_dir


def check_prior_images(class_images_dir, num_samples=200):
    """检查并验证先验图像"""
    if not os.path.exists(class_images_dir):
        return 0
        
    image_files = [f for f in os.listdir(class_images_dir) 
                  if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    return len(image_files)
