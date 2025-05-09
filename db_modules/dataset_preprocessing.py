"""
图像数据集预处理工具
提供验证、修复和预处理图像数据集的功能
"""

import os
import logging
from pathlib import Path
from PIL import Image, ImageOps, UnidentifiedImageError
import shutil

logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("DatasetPreprocessor")

class DatasetPreprocessor:
    def __init__(self, input_dir):
        """初始化数据集预处理器
        
        Args:
            input_dir: 输入图像目录
        """
        self.input_dir = input_dir
        self.valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff'}
        
    def validate_dataset(self, fix_problems=False):
        """验证数据集中的所有图像
        
        Args:
            fix_problems: 如果设为True，则尝试修复识别到的问题
            
        Returns:
            元组 (valid_images, invalid_images, fixed_images)
        """
        if not os.path.exists(self.input_dir):
            logger.error(f"输入目录不存在: {self.input_dir}")
            return [], [], []
            
        valid_images = []
        invalid_images = []
        fixed_images = []
        
        logger.info(f"开始验证数据集: {self.input_dir}")
        
        for root, _, files in os.walk(self.input_dir):
            for file in files:
                file_ext = os.path.splitext(file)[1].lower()
                if file_ext not in self.valid_extensions:
                    continue
                    
                image_path = os.path.join(root, file)
                is_valid, issue = self._validate_image(image_path)
                
                if is_valid:
                    valid_images.append(image_path)
                else:
                    if fix_problems:
                        fixed_path = self._try_fix_image(image_path, issue)
                        if fixed_path:
                            fixed_images.append(fixed_path)
                        else:
                            invalid_images.append((image_path, issue))
                    else:
                        invalid_images.append((image_path, issue))
        
        # 打印统计信息
        logger.info(f"验证完成: {len(valid_images)} 个有效图像, {len(invalid_images)} 个无效图像")
        
        if fix_problems:
            logger.info(f"修复尝试: {len(fixed_images)} 个图像已修复")
        
        if invalid_images:
            logger.warning("无效图像列表:")
            for path, issue in invalid_images[:10]:
                logger.warning(f" - {os.path.basename(path)}: {issue}")
            if len(invalid_images) > 10:
                logger.warning(f" ...以及其他 {len(invalid_images) - 10} 个问题图像")
        
        return valid_images, invalid_images, fixed_images
        
    def _validate_image(self, image_path):
        """验证单个图像
        
        Args:
            image_path: 图像文件路径
            
        Returns:
            元组 (is_valid, issue_description)
        """
        if not os.path.exists(image_path):
            return False, "文件不存在"
            
        if os.path.getsize(image_path) == 0:
            return False, "文件为空"
            
        try:
            with Image.open(image_path) as img:
                if not hasattr(img, 'width') or not hasattr(img, 'height'):
                    return False, "无法获取图像尺寸"
                    
                if img.width <= 0 or img.height <= 0:
                    return False, f"图像尺寸无效 (宽度={img.width}, 高度={img.height})"
                
                # 检查图像是否有实际内容
                return True, "图像有效"
        except UnidentifiedImageError:
            return False, "无法识别的图像格式"
        except Exception as e:
            return False, f"验证出错: {str(e)}"
    
    def _try_fix_image(self, image_path, issue):
        """尝试修复问题图像
        
        Args:
            image_path: 图像文件路径
            issue: 问题描述
            
        Returns:
            修复后的图像路径，如果无法修复则返回None
        """
        fixed_dir = os.path.join(os.path.dirname(image_path), "_fixed")
        os.makedirs(fixed_dir, exist_ok=True)
        
        basename = os.path.basename(image_path)
        fixed_path = os.path.join(fixed_dir, f"fixed_{basename}")
        
        try:
            # 尝试不同的修复方法
            if "无法识别的图像格式" in issue or "验证出错" in issue:
                # 可能的修复：重新保存为标准格式
                img = Image.open(image_path)
                img.save(fixed_path, format="PNG")
                logger.info(f"修复尝试 - 重新保存为PNG: {basename}")
                return fixed_path
                
            if "图像尺寸无效" in issue:
                # 对于EXIF旋转问题，尝试应用EXIF方向
                img = Image.open(image_path)
                img = ImageOps.exif_transpose(img)
                img.save(fixed_path, format="PNG")
                logger.info(f"修复尝试 - 应用EXIF旋转: {basename}")
                return fixed_path
                
            # 其他修复方法可以在这里添加
            
            return None  # 无法修复
        except Exception as e:
            logger.warning(f"修复失败 {basename}: {str(e)}")
            return None
    
    @staticmethod
    def clean_dataset(dataset_path):
        """清理数据集，移除或修复无效图像"""
        preprocessor = DatasetPreprocessor(dataset_path)
        valid, invalid, fixed = preprocessor.validate_dataset(fix_problems=True)
        
        if invalid:
            # 创建问题文件目录
            problem_dir = os.path.join(dataset_path, "_problem_files")
            os.makedirs(problem_dir, exist_ok=True)
            
            # 移动问题文件
            for problem_file, _ in invalid:
                target = os.path.join(problem_dir, os.path.basename(problem_file))
                try:
                    shutil.move(problem_file, target)
                    logger.info(f"已移动问题文件到: {target}")
                except Exception as e:
                    logger.error(f"移动文件失败 {problem_file}: {str(e)}")
            
            return len(valid), len(invalid), len(fixed)
        
        return len(valid), 0, len(fixed)

# 如果作为独立脚本运行
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="图像数据集预处理工具")
    parser.add_argument("--input_dir", type=str, required=True, help="输入图像目录")
    parser.add_argument("--fix", action="store_true", help="尝试修复问题图像")
    parser.add_argument("--clean", action="store_true", help="清理数据集，移除问题文件")
    
    args = parser.parse_args()
    
    if args.clean:
        valid_count, invalid_count, fixed_count = DatasetPreprocessor.clean_dataset(args.input_dir)
        print(f"数据集清理完成: {valid_count} 有效, {invalid_count} 无效, {fixed_count} 已修复")
    else:
        preprocessor = DatasetPreprocessor(args.input_dir)
        valid, invalid, fixed = preprocessor.validate_dataset(fix_problems=args.fix)
        
        print(f"数据集验证完成:")
        print(f" - 有效图像: {len(valid)}")
        print(f" - 无效图像: {len(invalid)}")
        if args.fix:
            print(f" - 已修复图像: {len(fixed)}")
        
        if invalid:
            print("\n问题文件:")
            for path, issue in invalid[:10]:
                print(f" - {os.path.basename(path)}: {issue}")
            if len(invalid) > 10:
                print(f" - ...以及其他 {len(invalid) - 10} 个问题文件")
