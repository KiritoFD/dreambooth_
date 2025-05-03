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
    optional_missing = []
    
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
    
    if optional_missing:
        print("\n以下可选依赖未安装，但可以帮助降低内存使用:")
        print(f"pip install {' '.join(optional_missing)}")
    
    return missing_deps

def create_debug_report(model_path, steps_completed, total_steps, error_message=None):
    """创建详细的调试报告，分析训练中断原因"""
    debug_report_path = os.path.join(model_path, "debug_report.txt")
    
    with open(debug_report_path, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("DreamBooth训练调试报告\n")
        f.write("=" * 80 + "\n\n")
        
        # 记录基本信息
        f.write(f"报告生成时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"训练进度: {steps_completed}/{total_steps} 步 ({steps_completed/total_steps*100:.1f}%)\n\n")
        
        # 系统信息
        f.write("系统信息:\n")
        f.write("-" * 40 + "\n")
        
        try:
            import platform
            f.write(f"操作系统: {platform.system()} {platform.version()}\n")
            f.write(f"Python版本: {platform.python_version()}\n")
        except:
            f.write("无法获取系统信息\n")
        
        # GPU信息
        f.write("\nGPU信息:\n")
        f.write("-" * 40 + "\n")
        
        if torch.cuda.is_available():
            try:
                gpu_name = torch.cuda.get_device_name(0)
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
                gpu_free = torch.cuda.memory_reserved(0) / 1024**3
                gpu_used = torch.cuda.memory_allocated(0) / 1024**3
                
                f.write(f"GPU型号: {gpu_name}\n")
                f.write(f"GPU总内存: {gpu_memory:.2f}GB\n")
                f.write(f"GPU已分配内存: {gpu_used:.2f}GB\n")
                f.write(f"GPU预留内存: {gpu_free:.2f}GB\n")
            except:
                f.write("无法获取完整GPU信息\n")
        else:
            f.write("未检测到GPU\n")
        
        # 检查检查点文件
        f.write("\n检查点文件分析:\n")
        f.write("-" * 40 + "\n")
        
        checkpoint_path = os.path.join(model_path, "checkpoint.pt")
        if os.path.exists(checkpoint_path):
            f.write(f"检查点文件存在: {checkpoint_path}\n")
            f.write(f"检查点文件大小: {os.path.getsize(checkpoint_path)/1024/1024:.2f}MB\n")
            
            try:
                checkpoint = torch.load(checkpoint_path)
                f.write(f"检查点包含的步数: {checkpoint.get('global_step', '未知')}\n")
                f.write(f"检查点内容: {', '.join(checkpoint.keys())}\n")
            except Exception as e:
                f.write(f"无法加载检查点文件: {str(e)}\n")
        else:
            f.write("未找到检查点文件\n")
        
        # 检查可能的错误日志
        error_log_path = os.path.join(model_path, "error_log.txt")
        if os.path.exists(error_log_path):
            f.write("\n错误日志内容:\n")
            f.write("-" * 40 + "\n")
            try:
                with open(error_log_path, "r") as error_file:
                    f.write(error_file.read())
            except:
                f.write("无法读取错误日志\n")
        
        # 记录当前错误信息
        if error_message:
            f.write("\n当前错误信息:\n")
            f.write("-" * 40 + "\n")
            f.write(str(error_message) + "\n")
        
        # 给出可能的原因分析和建议
        f.write("\n可能的中断原因分析:\n")
        f.write("-" * 40 + "\n")
        
        if steps_completed == 0:
            f.write("训练可能一开始就失败了，主要原因可能是:\n")
            f.write("1. 模型加载失败 - 检查模型路径和网络连接\n")
            f.write("2. 内存不足 - 尝试减少batch size或prior_images数量\n")
            f.write("3. 环境问题 - 检查CUDA版本与PyTorch版本的兼容性\n")
        elif steps_completed < total_steps * 0.1:  # 不到10%
            f.write("训练在早期阶段失败，主要原因可能是:\n")
            f.write("1. 内存逐渐耗尽 - 模型在训练过程中累积了过多内存\n")
            f.write("2. 数据问题 - 某些训练图像可能有问题\n")
            f.write("3. 优化器设置问题 - 学习率可能过高导致不稳定\n")
        else:
            f.write("训练在进行一段时间后中断，主要原因可能是:\n")
            f.write("1. 系统或用户中断 - 检查是否有系统休眠或电源管理问题\n")
            f.write("2. 长时间训练导致的硬件过热 - 检查GPU温度\n")
            f.write("3. 其他程序干扰 - 检查是否有其他程序抢占GPU资源\n")
        
        # 给出解决建议
        f.write("\n建议解决方案:\n")
        f.write("-" * 40 + "\n")
        f.write("1. 减少资源使用:\n")
        f.write("   - 降低prior_images数量 (--prior_images 5)\n")
        f.write("   - 不训练文本编码器 (移除--train_text_encoder)\n")
        f.write("   - 减小batch_size (--batch_size 1)\n")
        f.write("2. 恢复训练:\n")
        f.write("   - 使用--resume参数从上次中断处继续\n")
        f.write("3. 检查环境:\n")
        f.write("   - 更新GPU驱动\n")
        f.write("   - 确保CUDA版本与PyTorch兼容\n")
        f.write("   - 关闭其他占用GPU的程序\n")
    
    return debug_report_path

def detect_cuda_error_in_log(log_file):
    """检查日志文件中是否存在CUDA错误"""
    if not os.path.exists(log_file):
        return None
        
    try:
        with open(log_file, "r") as f:
            content = f.read()
            
        # 检查常见CUDA错误
        if "CUDA error: out of memory" in content:
            return "CUDA内存不足错误"
        elif "CUDA error" in content:
            return "CUDA错误"
        elif "CUDA kernel errors" in content:
            return "CUDA核心错误"
    except:
        pass
    
    return None

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

    # 尝试导入低内存辅助模块
    try:
        from low_memory_helper import get_optimal_settings, optimize_for_inference, low_memory_context
        HAS_LOW_MEMORY_HELPER = True
        print("已加载低内存训练辅助模块")
    except ImportError:
        HAS_LOW_MEMORY_HELPER = False
        print("未找到低内存训练辅助模块，将使用默认设置")

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
    
    # 检查并分析错误
    if args.check_errors:
        debug_report_path = os.path.join(args.model_path, "debug_report.txt")
        error_log_path = os.path.join(args.model_path, "error_log.txt")
        
        # 检查是否存在调试报告
        if os.path.exists(debug_report_path):
            print("\n发现调试报告，分析结果:")
            with open(debug_report_path, "r") as f:
                report = f.read()
                
            if "CUDA error: out of memory" in report:
                print("\n[分析结果] 训练失败原因：GPU显存不足")
                
                # 检查GPU并提供具体建议
                if torch.cuda.is_available() and HAS_LOW_MEMORY_HELPER:
                    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
                    config, recommendation = get_optimal_settings(gpu_memory)
                    
                    print(f"\n您的GPU ({torch.cuda.get_device_name(0)}) 有 {gpu_memory:.1f}GB 显存")
                    print(f"推荐配置: {recommendation}")
                    print("\n请使用以下命令重新开始训练:")
                    
                    cmd = f"python main.py --train --low_memory"
                    if not config.train_text_encoder:
                        cmd += " --prior_images {0}".format(config.prior_images_small)
                    
                    print(cmd)
                else:
                    print("\n建议：使用低内存模式并减少先验图像数量")
                    print("python main.py --train --low_memory --prior_images 5")
        
        print("\n完成错误分析")
        return 0
    
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
            
            # 检查是否需要低内存模式
            if gpu_memory < 10 and not args.low_memory:
                print(f"\n警告: 检测到GPU内存较小 ({gpu_memory:.1f}GB)")
                print("建议启用低内存模式: --low_memory")
            
            # 应用低内存优化
            memory_optimized = False
            if args.low_memory and HAS_LOW_MEMORY_HELPER:
                config, recommendation = get_optimal_settings(gpu_memory)
                print(f"\n已启用低内存模式: {recommendation}")
                
                # 根据配置修改参数
                if not config.train_text_encoder:
                    args.train_text_encoder = False
                    print("- 已禁用文本编码器训练以节省内存")
                
                if args.prior_images > config.prior_images_small:
                    args.prior_images = config.prior_images_small
                    print(f"- 已将先验图像数量降低至 {config.prior_images_small} 以节省内存")
                
                memory_optimized = True
            
            # 提供内存使用建议
            if gpu_memory < 6:
                print("\n警告: 您的GPU显存非常有限，强烈推荐以下设置:")
                print("1. 减少先验图像数量: --prior_images 5")
                print("2. 不训练文本编码器: 移除 --train_text_encoder")
                print("3. 减小训练步数: --steps 400")
            
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
            # 准备低内存训练参数
            if args.low_memory and HAS_LOW_MEMORY_HELPER:
                # 添加低内存优化参数
                additional_train_params = {
                    "attention_slice_size": config.attention_slice_size,
                    "gradient_checkpointing": config.gradient_checkpointing,
                    "use_8bit_adam": config.use_8bit_adam
                }
            else:
                additional_train_params = {}
            
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
                memory_mgr=memory_mgr,
                resume_training=args.resume,
                **additional_train_params  # 添加额外的低内存参数
            )
            
            # 检查训练是否正常完成所有步骤
            checkpoint_path = os.path.join(args.model_path, "checkpoint.pt")
            if os.path.exists(checkpoint_path):
                try:
                    checkpoint = torch.load(checkpoint_path)
                    completed_steps = checkpoint.get("global_step", 0)
                    completion_percentage = (completed_steps / args.steps) * 100
                    
                    if completed_steps < args.steps * 0.9:  # 未完成90%
                        print(f"\n训练提前终止: 仅完成了 {completed_steps}/{args.steps} 步 ({completion_percentage:.1f}%)")
                        print("可能的原因:")
                        print("1. 内存不足导致程序崩溃")
                        print("2. 训练过程中出现了错误")
                        print("3. 训练时间过长被系统或用户中断")
                        
                        # 创建详细调试报告
                        debug_report = create_debug_report(args.model_path, completed_steps, args.steps)
                        print(f"\n已生成详细的调试报告: {debug_report}")
                        print("请查看该文件以获取更多信息和解决方案建议")
                        
                        print("\n要继续训练，请使用: python main.py --train --resume --model_path", args.model_path)
                    else:
                        print(f"\n训练成功完成! 共执行了 {completed_steps}/{args.steps} 步 ({completion_percentage:.1f}%)")
                except Exception as e:
                    print(f"\n训练状态检查错误: {str(e)}")
                    debug_report = create_debug_report(args.model_path, 0, args.steps, error_message=e)
                    print(f"已生成错误调试报告: {debug_report}")
            else:
                print("\n未找到检查点文件，无法确定训练状态")
                debug_report = create_debug_report(args.model_path, 0, args.steps, 
                                                 error_message="未找到检查点文件，训练可能完全失败")
                print(f"已生成错误调试报告: {debug_report}")
        
        except KeyboardInterrupt:
            print("\n训练被用户手动中断")
            
            # 收集中断时的信息并生成报告
            checkpoint_path = os.path.join(args.model_path, "checkpoint.pt")
            completed_steps = 0
            if os.path.exists(checkpoint_path):
                try:
                    checkpoint = torch.load(checkpoint_path)
                    completed_steps = checkpoint.get("global_step", 0)
                except:
                    pass
            
            debug_report = create_debug_report(args.model_path, completed_steps, args.steps, 
                                             error_message="用户手动中断训练")
            print(f"\n已生成中断调试报告: {debug_report}")
            print("\n您可以使用 --resume 参数稍后恢复训练:")
            print(f"python main.py --train --resume --model_path {args.model_path}")
            
        except Exception as e:
            print(f"训练遇到错误: {str(e)}")
            import traceback
            error_details = traceback.format_exc()
            print(error_details)
            
            # 保存错误信息到日志
            os.makedirs(args.model_path, exist_ok=True)
            with open(os.path.join(args.model_path, "error_log.txt"), "w") as f:
                f.write(f"错误时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"错误信息: {str(e)}\n\n")
                f.write(error_details)
            
            if "CUDA out of memory" in str(e) or ("cuda" in str(e).lower() and "memory" in str(e).lower()):
                print("\n检测到内存不足错误！")
                print("\n请尝试以下解决方案:")
                print("1. 使用低内存模式重新训练: python main.py --train --low_memory")
                print("2. 如果已经使用低内存模式，进一步减少先验图像: --prior_images 3")
                print("3. 如果仍然存在问题，请检查是否有其他应用占用GPU内存")
                
            # 检查是否存在检查点，判断完成的步骤
            checkpoint_path = os.path.join(args.model_path, "checkpoint.pt")
            completed_steps = 0
            if os.path.exists(checkpoint_path):
                try:
                    checkpoint = torch.load(checkpoint_path)
                    completed_steps = checkpoint.get("global_step", 0)
                except:
                    pass
            
            # 生成详细调试报告
            debug_report = create_debug_report(args.model_path, completed_steps, args.steps, error_message=str(e))
            print(f"\n已生成错误调试报告: {debug_report}")
            
            if completed_steps > 0:
                print(f"\n可以从已完成的 {completed_steps} 步继续训练:")
                print(f"python main.py --train --resume --model_path {args.model_path}")
    
    # 处理推理任务
    elif args.infer:
        if not args.prompt:
            print("错误: 推理模式需要指定 --prompt 参数")
            return 1
            
        # 低内存推理优化
        if HAS_LOW_MEMORY_HELPER and args.low_memory:
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
