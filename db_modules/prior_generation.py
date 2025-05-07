"""
DreamBooth 先验图像生成模块
负责生成用于先验保留损失的类别图像
"""
import os
import torch
from tqdm.auto import tqdm


def generate_prior_images(
    pipeline, class_prompt, output_dir, num_samples=200,
    batch_size=None, theory_notes_enabled=False, theory_step_fn=None,
    use_local_models=False, local_model_path=None, class_images_dir=None
):
    """生成类别先验图像"""
    # 打印先验保留理论
    if theory_notes_enabled and theory_step_fn:
        theory = theory_step_fn("prior_preservation")
        if theory:
            from theory_notes import print_theory_step
            print_theory_step("2", theory["title"], theory["description"])
    
    # 使用指定的类别图像目录或创建默认目录
    if class_images_dir is None:
        class_images_dir = os.path.join(output_dir, "class_images")
        print(f"未指定类别图像目录，使用默认路径: {class_images_dir}")
    
    # 确保目录存在
    os.makedirs(class_images_dir, exist_ok=True)
    print(f"使用类别图像目录: {class_images_dir}")
    
    # 检查现有图像
    existing_images = len([f for f in os.listdir(class_images_dir) 
                         if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    if existing_images >= 1:
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
    
    # 添加错误处理
    try:
        num_batches = (num_samples + batch_size - 1) // batch_size
        
        # 如果使用本地模型，确保已正确加载
        if use_local_models and local_model_path and hasattr(pipeline, 'model_path'):
            print(f"使用本地模型路径: {pipeline.model_path}")
        
        # 批量生成图像
        for batch_idx in tqdm(range(num_batches), desc="生成类别图像"):
            batch_prompts = [class_prompt] * min(batch_size, num_samples - batch_idx * batch_size)
            with torch.no_grad():
                outputs = pipeline(batch_prompts, num_inference_steps=50, guidance_scale=7.5)
                
            for i, image in enumerate(outputs.images):
                img_idx = batch_idx * batch_size + i
                image.save(os.path.join(class_images_dir, f"class_{img_idx:04d}.png"))
        
        print(f"成功生成了 {num_samples} 张类别图像，已保存到 {class_images_dir}")
    except Exception as e:
        print(f"生成类别图像时出错: {str(e)}")
        print("尝试使用备用方法或减少生成数量...")
        
        # 如果目录为空，至少生成一些示例图像以防止训练失败
        if len(os.listdir(class_images_dir)) == 0:
            try:
                # 尝试使用降低的参数重新生成
                reduced_samples = min(20, num_samples)
                print(f"尝试生成较少的 {reduced_samples} 张图像...")
                outputs = pipeline([class_prompt] * reduced_samples, num_inference_steps=30, guidance_scale=7.0)
                for i, image in enumerate(outputs.images):
                    image.save(os.path.join(class_images_dir, f"class_{i:04d}.png"))
                print(f"成功生成了 {reduced_samples} 张类别图像")
            except Exception as inner_e:
                print(f"备用方法也失败: {str(inner_e)}")
                print("警告: 无法生成类别图像，先验保留功能可能无法正常工作")
    
    return class_images_dir


def check_prior_images(class_images_dir, num_samples=200):
    """检查并验证先验图像"""
    if not os.path.exists(class_images_dir):
        return 0
        
    image_files = [f for f in os.listdir(class_images_dir) 
                  if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    return len(image_files)
