import os
import gc
import torch
from diffusers import StableDiffusionPipeline

class MemoryManager:
    """内存管理辅助类"""
    def __init__(self, is_main_process=True):
        self.is_main_process = is_main_process
        self.peak_memory = 0
        self.initial_memory = self.get_memory_usage()
        
    def get_memory_usage(self):
        return torch.cuda.memory_allocated()/1024**2 if torch.cuda.is_available() else 0
    
    def cleanup(self, log_message=None):
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
    def log_memory_stats(self, message):
        if self.is_main_process and torch.cuda.is_available():
            current = self.get_memory_usage()
            self.peak_memory = max(self.peak_memory, current)
            print(f"{message}: {current:.2f}MB / 峰值: {self.peak_memory:.2f}MB")

def download_small_model():
    """自动下载小型模型，适合低资源设备"""
    print("选择适合低资源设备的小型模型...")
    
    # 定义小型模型选项
    small_models = [
        "CompVis/stable-diffusion-v1-4",  # 较旧但稳定的SD1.4，兼容性好
        "runwayml/stable-diffusion-v1-5", # 标准SD1.5
    ]
    
    # 选择默认小型模型
    chosen_model = small_models[0]
    print(f"已选择模型: {chosen_model}")
    
    return chosen_model

def check_dependencies():
    """检查必要的依赖项是否已安装"""
    missing_deps = []
    try:
        import torch
    except ImportError:
        missing_deps.append("torch")
    
    try:
        import accelerate
    except ImportError:
        missing_deps.append("accelerate")
        
    try:
        import transformers
    except ImportError:
        missing_deps.append("transformers")
        
    try:
        import diffusers
    except ImportError:
        missing_deps.append("diffusers")
    
    return missing_deps

def inference(model_path, prompt, num_images=1, output_image_path=None):
    """使用训练好的模型生成图像"""
    # 加载标识符
    identifier = None
    if os.path.exists(os.path.join(model_path, "identifier.txt")):
        with open(os.path.join(model_path, "identifier.txt"), "r") as f:
            identifier = f.read().strip()
    
    # 加载模型
    pipeline = StableDiffusionPipeline.from_pretrained(model_path)
    if torch.cuda.is_available():
        pipeline = pipeline.to("cuda")
    
    # 生成图像
    print(f"使用提示词: {prompt}")
    images = pipeline(
        prompt,
        num_images_per_prompt=num_images,
        num_inference_steps=50,
        guidance_scale=7.5
    ).images
    
    # 保存图像
    os.makedirs("outputs", exist_ok=True)
    for i, image in enumerate(images):
        if output_image_path and i == 0:
            image.save(output_image_path)
            print(f"图像已保存到: {output_image_path}")
        else:
            import datetime
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = f"outputs/generated_{timestamp}_{i}.png"
            image.save(save_path)
            print(f"图像已保存到: {save_path}")
    
    return images
