import os
import gc
import torch
import argparse
import datetime
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
        with open(os.path.join(model_path, "identifier.txt")) as f:
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
    optional_missing = []
    version_issues = []
    
    # 检查核心依赖
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
        # 检查transformers版本 - 某些版本可能存在tokenizer问题
        transformers_version = getattr(transformers, '__version__', '0.0.0')
        if transformers_version < '4.20.0':
            version_issues.append(f"transformers=={transformers_version} (建议 >= 4.20.0)")
    except ImportError:
        missing_deps.append("transformers")
    
    try:
        import diffusers
    except ImportError:
        missing_deps.append("diffusers")
    
    # 检查可选内存优化依赖
    try:
        import bitsandbytes
    except ImportError:
        optional_missing.append("bitsandbytes")
    
    try:
        import xformers
    except ImportError:
        optional_missing.append("xformers")
        
    # 检查tokenizer相关依赖
    try:
        from transformers import CLIPTokenizer
    except ImportError:
        optional_missing.append("transformers[tokenizers]")
    
    if version_issues:
        print("\n检测到潜在的版本兼容性问题:")
        for issue in version_issues:
            print(f"- {issue}")
        print("建议运行: pip install --upgrade transformers")
    
    if optional_missing:
        print("\n以下可选依赖未安装，但可以帮助降低内存使用或提高兼容性:")
        print(f"pip install {' '.join(optional_missing)}")
    
    return missing_deps

def main():
    # 分析提前退出原因
    print("\n【DreamBooth训练工具】")
    print("版本: 1.0.0")
    
    # 检查依赖项
    missing = check_dependencies()
    if missing:
        print("缺少必要的依赖项，请安装:")
        print(f"pip install {' '.join(missing)}")
        return 1

    # 尝试导入低内存辅助模块
    try:
        from db_modules.memory_optimization import get_optimal_settings, optimize_for_inference
        HAS_MEMORY_OPTIMIZATION = True
        print("已加载内存优化模块")
    except ImportError:
        HAS_MEMORY_OPTIMIZATION = False
        print("未找到内存优化模块，将使用标准内存设置")

    # 尝试导入调试模块
    try:
        from db_modules.debugging import analyze_training_failure, create_debug_report
        HAS_DEBUGGING = True
    except ImportError:
        HAS_DEBUGGING = False
    
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
    parser.add_argument("--steps", type=int, default=1000, help="训练步数")
    parser.add_argument("--prior_weight", type=float, default=1.0, help="先验保留损失权重")
    parser.add_argument("--train_text_encoder", action="store_true", help="是否训练文本编码器")
    parser.add_argument("--prior_images", type=int, default=10, help="先验保留的类别图像数量")
    parser.add_argument("--batch_size", type=int, default=1, help="训练批次大小")
    parser.add_argument("--memory_efficient", action="store_true", help="启用内存高效模式")
    parser.add_argument("--low_memory", action="store_true", help="启用低内存模式")
    parser.add_argument("--check_errors", action="store_true", help="检查并分析错误")
    
    args = parser.parse_args()
    
    # 处理训练任务
    if args.train:
        # 内存优化逻辑
        low_memory_params = {}
        if torch.cuda.is_available():
            gpu_info = torch.cuda.get_device_name()
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"GPU: {gpu_info} ({gpu_memory:.1f}GB)")
            
            # 低内存模式自动检测和建议
            if args.low_memory or gpu_memory < 10:
                if HAS_MEMORY_OPTIMIZATION:
                    config = get_optimal_settings(gpu_memory)
                    print(f"\n已应用内存优化: 针对{gpu_memory:.1f}GB GPU")
                    
                    # 应用优化参数
                    if not args.train_text_encoder and config.get("disable_text_encoder_training", False):
                        args.train_text_encoder = False
                        print("- 已禁用文本编码器训练")
                    
                    prior_suggestion = config.get("prior_images", args.prior_images)
                    if args.prior_images > prior_suggestion:
                        args.prior_images = prior_suggestion
                        print(f"- 已优化先验图像数量: {prior_suggestion}")
                    
                    # 设置低内存训练参数
                    low_memory_params = {
                        "low_memory": True,
                        "attention_slice_size": config.get("attention_slice_size", 0),
                        "gradient_checkpointing": config.get("gradient_checkpointing", True),
                        "use_8bit_adam": config.get("use_8bit_adam", True)
                    }
        
        # 清理内存
        memory_mgr = MemoryManager()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # 开始训练
        print(f"开始训练，步数: {args.steps}")
        try:
            identifier, training_successful = dreambooth_training(
                pretrained_model_name_or_path=args.model_name or download_small_model(),
                instance_data_dir=args.instance_data_dir,
                output_dir=args.model_path,
                class_prompt=args.class_prompt,
                max_train_steps=args.steps,
                prior_preservation_weight=args.prior_weight,
                train_text_encoder=args.train_text_encoder,
                prior_generation_samples=args.prior_images,
                train_batch_size=args.batch_size,
                memory_mgr=memory_mgr,
                resume_training=args.resume,
                **low_memory_params
            )
            
            # 训练完成后的操作
            if training_successful:
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
            else:
                print("\n训练未成功完成。")
                if HAS_DEBUGGING:
                    print("正在生成调试报告...")
                    debug_path = create_debug_report(args.model_path, args.steps)
                    print(f"调试报告已保存至: {debug_path}")
                    print("可以使用 --resume 参数继续中断的训练")
        
        except Exception as e:
            print(f"训练遇到错误: {str(e)}")
            import traceback
            traceback.print_exc()
            
            # 错误处理和调试
            if HAS_DEBUGGING:
                analyze_training_failure(e, args.model_path, args.steps)
    
    # 处理推理任务
    elif args.infer:
        if not args.prompt:
            print("错误: 推理模式需要指定 --prompt 参数")
            return 1
            
        # 低内存推理优化
        if HAS_MEMORY_OPTIMIZATION and args.low_memory:
            print("使用低内存模式进行推理...")
            try:
                from diffusers import StableDiffusionPipeline
                
                # 加载模型
                pipeline = StableDiffusionPipeline.from_pretrained(args.model_path)
                
                # 应用低内存优化
                pipeline = optimize_for_inference(pipeline)
                
                # 创建自定义推理函数
                def optimized_inference(prompt, num_images=1, output_path=None):
                    with torch.inference_mode():
                        with torch.cuda.amp.autocast():
                            outputs = pipeline(
                                prompt,
                                num_images_per_prompt=num_images,
                                num_inference_steps=30 if args.low_memory else 50,
                                guidance_scale=7.5
                            )
                    
                    # 保存图像
                    os.makedirs("outputs", exist_ok=True)
                    for i, image in enumerate(outputs.images):
                        if output_path and i == 0:
                            image.save(output_path)
                            print(f"图像已保存到: {output_path}")
                        else:
                            import datetime
                            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                            save_path = f"outputs/generated_{timestamp}_{i}.png"
                            image.save(save_path)
                            print(f"图像已保存到: {save_path}")
                    
                    return outputs.images
                
                # 使用优化后的推理
                optimized_inference(args.prompt)
                return 0
            except Exception as e:
                print(f"低内存优化推理失败: {e}")
                print("回退到标准推理...")
        
        inference(args.model_path, args.prompt)
    
    else:
        print("请指定 --train 或 --infer 参数")
        print("示例: python main.py --train --instance_data_dir ./my_images --class_prompt \"a cat\"")
    
    return 0

if __name__ == "__main__":
    exit(main())
