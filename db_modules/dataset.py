import os
import logging
import numpy as np
from pathlib import Path
from PIL import Image, ImageFile, UnidentifiedImageError
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import random

# 允许加载截断的图像文件 - 这可能会解决一些格式问题
ImageFile.LOAD_TRUNCATED_IMAGES = True

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class DreamBoothDataset(Dataset):
    def _validate_and_fix_image(self, image_path):
        """验证图像并尝试修复问题"""
        if not os.path.exists(image_path):
            return None, f"图像文件不存在: {image_path}"
            
        if os.path.getsize(image_path) == 0:
            return None, f"图像文件为空: {image_path}"
            
        try:
            with Image.open(image_path) as img:
                # 强制加载图像数据，以验证它实际上可以被加载
                img.load()
                
                # 检查图像尺寸
                if not hasattr(img, 'width') or not hasattr(img, 'height'):
                    return None, f"无法获取图像尺寸: {image_path}"
                    
                if img.width <= 0 or img.height <= 0:
                    # 尝试修复通过旋转元数据可能引起的0尺寸问题
                    try:
                        from PIL import ImageOps
                        img = ImageOps.exif_transpose(img)
                        if img.width <= 0 or img.height <= 0:
                            return None, f"无法修复无效尺寸: {image_path}"
                    except:
                        return None, f"图像尺寸无效 (宽度={img.width}, 高度={img.height}): {image_path}"
                
                # 确保是RGB模式，避免灰度或其他模式问题
                if img.mode != "RGB":
                    img = img.convert("RGB")
                
                # 返回已加载的图像
                return img, "图像有效"
        except UnidentifiedImageError:
            return None, f"无法识别的图像格式: {image_path}"
        except Exception as e:
            return None, f"图像验证过程中出错: {str(e)}"
    
    def _load_images(self, images_path, verbose=True):
        """加载图像文件，并增强错误处理和报告"""
        result = []
        skipped = []
        image_sources = []  # 跟踪每张图片来源便于调试
        
        if verbose:
            logger.debug(f"[DreamBoothDataset._load_images DEBUG] 从路径加载图像: {images_path}")
        
        if images_path is None:
            logger.error("[DreamBoothDataset._load_images ERROR] 图像路径为空!")
            return result
        
        if os.path.isfile(images_path):
            # 单个文件处理
            img, message = self._validate_and_fix_image(images_path)
            if img is not None:
                result.append(img)
                image_sources.append(images_path)
            else:
                skipped.append((images_path, message))
                if verbose:
                    logger.warning(f"[DreamBoothDataset._load_images WARNING] 无法加载或处理图像 {os.path.basename(images_path)}: {message}")
        elif os.path.isdir(images_path):
            # 目录处理
            valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff'}
            for file in sorted(os.listdir(images_path)):  # 排序确保加载顺序一致
                file_ext = os.path.splitext(file)[1].lower()
                if file_ext not in valid_extensions:
                    continue
                    
                image_path = os.path.join(images_path, file)
                img, message = self._validate_and_fix_image(image_path)
                
                if img is not None:
                    result.append(img)
                    image_sources.append(image_path)
                else:
                    skipped.append((image_path, message))
                    if verbose:
                        logger.warning(f"[DreamBoothDataset._load_images WARNING] 无法加载或处理图像 {file} 从 {images_path}: {message}")
        
        # 详细报告结果
        if verbose:
            logger.debug(f"[DreamBoothDataset._load_images DEBUG] 找到 {len(result) + len(skipped)} 个潜在图像文件，成功加载 {len(result)} 个图像从 {images_path}。")
            
            if result:
                # 报告成功加载的图像的尺寸
                sizes = [(img.width, img.height) for img in result[:5]]  # 仅取前5个样本
                logger.debug(f"[DreamBoothDataset._load_images DEBUG] 样本图像尺寸: {sizes}")
            
            if not result:
                logger.error(f"[DreamBoothDataset._load_images ERROR] 无法从 {images_path} 加载任何有效图像!")
                if skipped:
                    logger.error(f"[DreamBoothDataset._load_images ERROR] 所有 {len(skipped)} 个图像均无法加载.")
                    logger.error(f"[DreamBoothDataset._load_images ERROR] 常见问题包括: 文件损坏、格式不受支持或零尺寸错误。")
                    logger.error(f"[DreamBoothDataset._load_images ERROR] 尝试使用 'tools/fix_dataset.py --input_dir {images_path} --mode fix' 修复数据集。")
            
        return result, image_sources, skipped
    
    def __init__(self, instance_data_root, instance_prompt, tokenizer, 
                 size=512, center_crop=False, use_template=False,
                 template_file=None, overwrite_prompts=False,
                 text_encoder_type="SDv1", device="cuda", transform_type=None,
                 class_data_root=None, class_prompt=None, debug=True):
        self.size = size
        self.center_crop = center_crop
        self.tokenizer = tokenizer
        self.device = device  # 保存设备以便在__getitem__中使用
        self.text_encoder_type = text_encoder_type
        self.debug = debug
        
        # 打印调试信息
        if debug:
            logger.debug(f"[DreamBoothDataset DEBUG] Initializing dataset.")
            logger.debug(f"[DreamBoothDataset DEBUG] Received instance_images_path: '{instance_data_root}'")
        
        self.instance_images = []
        self.instance_sources = []
        self.class_images = []
        self.class_sources = []
        
        # 加载实例图像
        if instance_data_root:
            if os.path.exists(instance_data_root):
                if debug:
                    logger.debug(f"[DreamBoothDataset DEBUG] Instance images path exists. Attempting to load images...")
                self.instance_images, self.instance_sources, instance_skipped = self._load_images(instance_data_root)
            else:
                logger.warning(f"[DreamBoothDataset WARNING] Instance images path {instance_data_root} does not exist.")
        
        # 加载类别图像(如果有)
        if class_data_root and class_prompt:
            if os.path.exists(class_data_root):
                if debug:
                    logger.debug(f"[DreamBoothDataset DEBUG] Class images path exists. Attempting to load images...")
                self.class_images, self.class_sources, class_skipped = self._load_images(class_data_root)
            else:
                logger.warning(f"[DreamBoothDataset WARNING] Class images path {class_data_root} does not exist.")
        
        # 如果图像加载失败，提供帮助信息
        if not self.instance_images and instance_data_root:
            if self.debug:
                logger.error(f"[DreamBoothDataset ERROR] 无法加载任何实例图像! 请检查图像格式和路径: {instance_data_root}")
                logger.error("[DreamBoothDataset ERROR] 建议使用以下工具修复数据集:")
                logger.error(f"    python tools/fix_dataset.py --input_dir {instance_data_root} --mode fix")
        
        # 设置图像变换
        self._setup_image_transforms()
        
        # 生成提示
        self.instance_prompt = instance_prompt
        self.class_prompt = class_prompt
        self._setup_prompts()
        
        # 检查加载状态
        self.num_instance_images = len(self.instance_images)
        self.num_class_images = len(self.class_images)

        if debug:
            logger.debug(f"[DreamBoothDataset DEBUG] Dataset initialized with {self.num_instance_images} instance images and {self.num_class_images} class images.")
    
    def _setup_image_transforms(self):
        """设置图像转换管道"""
        # 确保尺寸是有效值
        if not self.size or self.size <= 0:
            # 对于SDXL推荐1024x1024
            if self.text_encoder_type and "sdxl" in self.text_encoder_type.lower():
                self.size = 1024
            else:
                self.size = 512  # 默认为标准SD
        
        # 创建转换管道
        if self.center_crop:
            self.image_transforms = transforms.Compose([
                transforms.Resize(self.size, interpolation=transforms.InterpolationMode.LANCZOS),
                transforms.CenterCrop(self.size),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ])
        else:
            self.image_transforms = transforms.Compose([
                transforms.Resize((self.size, self.size), interpolation=transforms.InterpolationMode.LANCZOS),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ])

    def _setup_prompts(self):
        """设置训练提示"""
        self.tokenized_instance_prompt = self.tokenizer(
            self.instance_prompt,
            padding="do_not_pad",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
        ).input_ids
        
        self.tokenized_class_prompt = None
        if self.class_prompt:
            self.tokenized_class_prompt = self.tokenizer(
                self.class_prompt,
                padding="do_not_pad",
                truncation=True,
                max_length=self.tokenizer.model_max_length,
            ).input_ids
            
    def __len__(self):
        return self.num_instance_images + self.num_class_images
    
    def __getitem__(self, index):
        example = {}
        
        if index < self.num_instance_images:
            example["instance_images"] = self.instance_images[index]
            example["instance_prompt_ids"] = self.tokenized_instance_prompt
            example["is_instance"] = True
            
            if self.debug and index == 0:  # 仅对第一个图像进行详细日志记录
                logger.debug(f"[DreamBoothDataset DEBUG] 加载实例图像 {self.instance_sources[index] if self.instance_sources else 'unknown'}")
                logger.debug(f"[DreamBoothDataset DEBUG] 原始尺寸: {example['instance_images'].size}")
        else:
            example["instance_images"] = self.class_images[index - self.num_instance_images]
            example["instance_prompt_ids"] = self.tokenized_class_prompt
            example["is_instance"] = False
            
            if self.debug and index == self.num_instance_images:  # 仅对第一个类别图像进行详细日志记录
                logger.debug(f"[DreamBoothDataset DEBUG] 加载类别图像 {self.class_sources[index - self.num_instance_images] if self.class_sources else 'unknown'}")
                logger.debug(f"[DreamBoothDataset DEBUG] 原始尺寸: {example['instance_images'].size}")
        
        # 应用图像变换
        if self.image_transforms is not None:
            try:
                example["pixel_values"] = self.image_transforms(example["instance_images"])
                
                # 明确将张量移到指定设备
                target_device = self.device if torch.cuda.is_available() else "cpu"
                example["pixel_values"] = example["pixel_values"].to(target_device)
                
                # 确保is_instance也在正确设备上
                example["is_instance"] = torch.tensor(example["is_instance"], 
                                                      dtype=torch.bool,
                                                      device=target_device)
                
            except Exception as e:
                if self.debug:
                    source = self.instance_sources[index] if index < self.num_instance_images else self.class_sources[index - self.num_instance_images]
                    if source:
                        logger.error(f"[DreamBoothDataset ERROR] 转换图像失败 {source}: {str(e)}")
                    else:
                        logger.error(f"[DreamBoothDataset ERROR] 转换图像失败 索引 {index}: {str(e)}")
                # 在错误情况下，创建黑色图像避免崩溃
                target_device = self.device if torch.cuda.is_available() else "cpu"
                example["pixel_values"] = torch.zeros((3, self.size, self.size), 
                                                    device=target_device)
                example["is_instance"] = torch.tensor(example["is_instance"], 
                                                     dtype=torch.bool,
                                                     device=target_device)
                
        # 不再需要原始PIL图像
        if "instance_images" in example:
            del example["instance_images"]
            
        return example