import os
import gc
import torch
import argparse
from dreambooth import dreambooth_training

class MemoryManager:
    """内存管理辅助类，精简版"""
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
    small_models = ["CompVis/stable-diffusion-v1-4"]
    chosen_model = small_models[0]
    print(f"已选择模型: {chosen_model}")
    return chosen_model

def inference(model_path, prompt, num_images=1, output_image_path=None):
    """使用训练好的模型生成图像"""
    from diffusers import StableDiffusionPipeline
    
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

def main():
    # 分析提前退出原因
    print("\n【训练提前终止分析工具】")
    print("以下是可能导致训练提前停止的原因:")
    print("1. 内存不足 - GPU显存耗尽导致OOM错误")
    print("2. 训练过程出现错误 - 通常与特定批次的数据相关")
    print("3. 用户手动中断 - 按下Ctrl+C或关闭终端")
    print("4. 系统不稳定 - 例如电源问题、系统过热等")
    
    # 检查依赖项
    missing = check_dependencies()
    if missing:
        print("缺少必要的依赖项，请安装:")
        print(f"pip install {' '.join(missing)}")
        return 1

    # 解析命令行参数
    parser = argparse.ArgumentParser(description="DreamBooth 训练和推理")
    parser.add_argument("--train", action="store_true", help="训练模式")
    parser.add_argument("--resume", action="store_true", help="从检查点恢复训练")
    parser.add_argument("--infer", action="store_true", help="推理模式")
    parser.add_argument("--model_name", type=str, default=None, help="预训练模型名称")
    parser.add_argument("--model_path", type=str, default="./output", help="模型路径")
    parser.add_argument("--instance_data_dir", type=str, default="./instance_images", help="实例图像目录")
    parser.add_argument("--class_prompt", type=str, default="a dog", help="类别提示词")
    parser.add_argument("--prompt", type=str, help="推理时的提示词")
    parser.add_argument("--steps", type=int, default=800, help="训练步数")
    parser.add_argument("--prior_weight", type=float, default=1.0, help="先验保留损失权重")
    parser.add_argument("--train_text_encoder", action="store_true", help="是否训练文本编码器")
    parser.add_argument("--prior_images", type=int, default=10, help="先验保留的类别图像数量")
    parser.add_argument("--batch_size", type=int, default=1, help="训练批次大小")
    parser.add_argument("--memory_efficient", action="store_true", help="启用内存高效模式")
    
    args = parser.parse_args()
    
    # 默认使用小型模型
    if args.model_name is None:
        args.model_name = download_small_model()
    
    # 处理训练任务
    if args.train:
        # 检查GPU
        if torch.cuda.is_available():
            gpu_info = torch.cuda.get_device_name()
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"GPU: {gpu_info} ({gpu_memory:.1f}GB)")
            
            # 提供内存使用建议
            if gpu_memory < 6:
                print("警告: 您的GPU显存较小，建议采用以下措施:")
                print("1. 减少先验图像数量: --prior_images 5")
                print("2. 不训练文本编码器: 移除 --train_text_encoder")
                print("3. 启用内存高效模式: --memory_efficient")
        else:
            print("警告: 未检测到GPU，训练将非常慢!")
            
        # 确保创建目录
        os.makedirs(args.instance_data_dir, exist_ok=True)
        os.makedirs(args.model_path, exist_ok=True)
        
        # 检查图像目录
        if os.path.exists(args.instance_data_dir):
            image_files = [f for f in os.listdir(args.instance_data_dir) 
                          if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            if not image_files:
                print(f"警告: 实例图像目录 '{args.instance_data_dir}' 中没有找到图像文件")
                print("请将要训练的图像放入此目录，然后重新运行")
                return 1
            else:
                print(f"找到 {len(image_files)} 张实例图像")
                
                # 如果图像过多，给出警告
                if len(image_files) > 10:
                    print("警告: 实例图像较多。DreamBooth通常使用3-5张图像效果最佳")
        
        # 清理内存
        memory_mgr = MemoryManager()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # 执行训练
        print(f"开始训练，步数: {args.steps}")
        try:
            identifier = dreambooth_training(
                pretrained_model_name_or_path=args.model_name,
                instance_data_dir=args.instance_data_dir,
                output_dir=args.model_path,
                class_prompt=args.class_prompt,
                max_train_steps=args.steps,
                prior_preservation_weight=args.prior_weight,
                train_text_encoder=args.train_text_encoder,
                prior_generation_samples=args.prior_images,
                train_batch_size=args.batch_size,
                memory_mgr=memory_mgr
            )
            
            print("\n训练成功完成!")
            
            # 询问是否测试
            if torch.cuda.is_available():
                test = input("是否测试模型效果? [y/n]: ").lower() == 'y'
                if test:
                    class_name = args.class_prompt.replace("a ", "").strip()
                    test_prompt = input(f"请输入提示词 (默认: a {identifier} {class_name} in a garden): ")
                    if not test_prompt:
                        test_prompt = f"a {identifier} {class_name} in a garden"
                    inference(args.model_path, test_prompt, 1, "./test_result.png")
        
        except KeyboardInterrupt:
            print("\n训练被用户中断")
            print("您可以使用 --resume 参数稍后恢复训练")
            
        except Exception as e:
            print(f"训练遇到错误: {str(e)}")
            import traceback
            traceback.print_exc()
            print("\n如果是内存不足错误，请尝试:")
            print("1. 减少先验图像数量: --prior_images 5")
            print("2. 不训练文本编码器: 移除 --train_text_encoder")
    
    # 处理推理任务
    elif args.infer:
        if not args.prompt:
            print("错误: 推理模式需要指定 --prompt 参数")
            return 1
            
        inference(args.model_path, args.prompt)
    
    else:
        print("请指定 --train 或 --infer 参数")
        print("示例: python main.py --train --instance_data_dir ./my_images --class_prompt \"a cat\"")
    
    return 0

if __name__ == "__main__":
    exit(main())
