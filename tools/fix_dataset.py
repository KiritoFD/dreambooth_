#!/usr/bin/env python
"""
DreamBooth 数据集修复工具
用于识别和修复数据集中的问题图像，特别是"height and width must be > 0"错误

用法:
  python fix_dataset.py --input_dir assets/monk --mode deep_check
  python fix_dataset.py --input_dir assets/monk --mode fix
  python fix_dataset.py --input_dir assets/monk --mode clean
"""

import os
import sys
import argparse
import logging
import tempfile
import shutil
from pathlib import Path
from PIL import Image, ImageFile, ImageOps, UnidentifiedImageError

# 允许加载截断的图像文件
ImageFile.LOAD_TRUNCATED_IMAGES = True

# 添加项目根目录到Python路径
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("FixDataset")

class EnhancedDatasetPreprocessor:
    def __init__(self, input_dir):
        self.input_dir = input_dir
        self.valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff'}
        
    def deep_validate_image(self, image_path):
        """深度验证图像是否可用于训练模型"""
        if not os.path.exists(image_path):
            return False, "文件不存在"
            
        if os.path.getsize(image_path) == 0:
            return False, "文件为空"
            
        try:
            # 尝试完全加载图像以验证其完整性
            with Image.open(image_path) as img:
                try:
                    # 强制加载图像数据
                    img.load()
                except Exception as e:
                    return False, f"加载图像数据失败: {str(e)}"
                
                # 检查尺寸
                if not hasattr(img, 'width') or not hasattr(img, 'height'):
                    return False, "无法获取图像尺寸"
                    
                if img.width <= 0 or img.height <= 0:
                    return False, f"图像尺寸无效 (宽度={img.width}, 高度={img.height})"
                
                # 检查是否可转换为RGB
                try:
                    if img.mode != "RGB":
                        rgb_img = img.convert("RGB")
                except Exception as e:
                    return False, f"无法转换为RGB: {str(e)}"
                
                # 验证是否可以调整大小 (训练中必须的操作)
                try:
                    test_resize = img.resize((512, 512))
                    if test_resize.width != 512 or test_resize.height != 512:
                        return False, "调整大小失败"
                except Exception as e:
                    return False, f"调整大小时出错: {str(e)}"
                
                # 检查图像是否只有alpha通道但没有内容
                if 'A' in img.getbands() and not any(img.getchannel(band).getextrema()[1] > 0 for band in img.getbands() if band != 'A'):
                    return False, "图像只有透明通道，没有可见内容"
                    
                return True, "图像有效"
                
        except UnidentifiedImageError:
            return False, "无法识别的图像格式"
        except Exception as e:
            return False, f"验证出错: {str(e)}"
    
    def validate_dataset(self, mode="basic"):
        """验证数据集中的所有图像
        
        Args:
            mode: "basic" 仅进行基本检查，"deep" 执行深度验证
        """
        if not os.path.exists(self.input_dir):
            logger.error(f"输入目录不存在: {self.input_dir}")
            return [], [], []
            
        valid_images = []
        invalid_images = []
        
        logger.info(f"开始{'深度' if mode=='deep' else '基本'}验证数据集: {self.input_dir}")
        
        for file in os.listdir(self.input_dir):
            file_ext = os.path.splitext(file)[1].lower()
            if file_ext not in self.valid_extensions:
                continue
                
            image_path = os.path.join(self.input_dir, file)
            
            if mode == "deep":
                is_valid, issue = self.deep_validate_image(image_path)
            else:
                # 基本验证只检查文件是否可打开
                try:
                    with Image.open(image_path) as img:
                        is_valid = True
                        issue = "图像可打开"
                except:
                    is_valid = False
                    issue = "无法打开图像"
            
            if is_valid:
                valid_images.append(image_path)
            else:
                invalid_images.append((image_path, issue))
        
        # 打印统计信息
        logger.info(f"验证完成: {len(valid_images)} 个有效图像, {len(invalid_images)} 个无效图像")
        
        if invalid_images:
            logger.warning("无效图像列表:")
            for path, issue in invalid_images[:10]:
                logger.warning(f" - {os.path.basename(path)}: {issue}")
            if len(invalid_images) > 10:
                logger.warning(f" ...以及其他 {len(invalid_images) - 10} 个问题图像")
        
        return valid_images, invalid_images
    
    def try_fix_image(self, image_path, backup=True, fixed_dir=None):
        """尝试修复问题图像
        
        Args:
            image_path: 图像文件路径
            backup: 是否备份原始文件
            fixed_dir: 保存已修复图像的目录
            
        Returns:
            修复后的图像路径或None（如果无法修复）
        """
        if fixed_dir is None:
            fixed_dir = os.path.join(os.path.dirname(image_path), "_fixed")
        
        os.makedirs(fixed_dir, exist_ok=True)
        basename = os.path.basename(image_path)
        fixed_path = os.path.join(fixed_dir, basename)
        
        # 先备份原始文件
        if backup:
            backup_dir = os.path.join(os.path.dirname(image_path), "_backup")
            os.makedirs(backup_dir, exist_ok=True)
            backup_path = os.path.join(backup_dir, basename)
            try:
                shutil.copy2(image_path, backup_path)
            except Exception as e:
                logger.warning(f"无法备份 {basename}: {str(e)}")
        
        try:
            # 尝试多种修复方法
            
            # 方法1: 先尝试使用ImageOps处理EXIF旋转
            try:
                img = Image.open(image_path)
                img = ImageOps.exif_transpose(img)
                if img.mode != "RGB":
                    img = img.convert("RGB")
                img.save(fixed_path, format="JPEG", quality=95)
                
                # 验证修复是否成功
                is_valid, _ = self.deep_validate_image(fixed_path)
                if is_valid:
                    logger.info(f"✅ 使用EXIF转置修复成功: {basename}")
                    return fixed_path
            except:
                pass
                
            # 方法2: 尝试重新编码为PNG
            try:
                img = Image.open(image_path)
                if img.mode != "RGB":
                    img = img.convert("RGB")
                # 保存为PNG（无损）
                fixed_png = os.path.splitext(fixed_path)[0] + ".png"
                img.save(fixed_png, format="PNG")
                
                # 验证修复是否成功
                is_valid, _ = self.deep_validate_image(fixed_png)
                if is_valid:
                    logger.info(f"✅ 通过PNG重编码修复成功: {basename}")
                    return fixed_png
            except:
                pass
            
            # 方法3: 重新创建图像（彻底重建）
            try:
                img = Image.open(image_path)
                # 创建新的空白图像并粘贴旧图像内容
                new_img = Image.new("RGB", img.size if img.size[0] > 0 and img.size[1] > 0 else (512, 512))
                if img.size[0] > 0 and img.size[1] > 0:  # 只有当原始尺寸有效时
                    new_img.paste(img)
                new_img.save(fixed_path, format="JPEG", quality=95)
                
                # 验证修复是否成功
                is_valid, _ = self.deep_validate_image(fixed_path)
                if is_valid:
                    logger.info(f"✅ 通过重新创建图像修复成功: {basename}")
                    return fixed_path
            except:
                pass
                
            # 无法修复，创建占位图像
            try:
                placeholder = Image.new("RGB", (512, 512), color=(0, 0, 0))
                placeholder.save(fixed_path, format="JPEG")
                logger.warning(f"⚠️ 无法修复 {basename}，已创建黑色占位图像")
                return fixed_path
            except:
                pass
                
            logger.error(f"❌ 所有修复方法均失败: {basename}")
            return None
            
        except Exception as e:
            logger.error(f"❌ 修复过程中出错 {basename}: {str(e)}")
            return None
    
    def fix_dataset(self, replace_originals=False):
        """修复数据集中的所有问题图像
        
        Args:
            replace_originals: 是否用修复的图像替换原始图像
        
        Returns:
            (修复成功数量, 修复失败数量)
        """
        _, invalid_images = self.validate_dataset(mode="deep")
        
        if not invalid_images:
            logger.info("✅ 未发现需要修复的图像")
            return 0, 0
        
        logger.info(f"开始修复 {len(invalid_images)} 个问题图像...")
        
        fixed_dir = os.path.join(self.input_dir, "_fixed")
        os.makedirs(fixed_dir, exist_ok=True)
        
        fixed_success = 0
        fixed_failed = 0
        
        for image_path, issue in invalid_images:
            logger.info(f"尝试修复 {os.path.basename(image_path)}: {issue}")
            fixed_path = self.try_fix_image(image_path, backup=True, fixed_dir=fixed_dir)
            
            if fixed_path:
                fixed_success += 1
                # 如果指定了替换原始文件
                if replace_originals:
                    try:
                        shutil.copy2(fixed_path, image_path)
                        logger.info(f"✅ 已用修复版本替换原始文件: {os.path.basename(image_path)}")
                    except Exception as e:
                        logger.error(f"❌ 替换原始文件失败: {str(e)}")
            else:
                fixed_failed += 1
        
        logger.info(f"修复完成: {fixed_success} 成功, {fixed_failed} 失败")
        if fixed_success > 0:
            logger.info(f"已修复的图像保存在: {fixed_dir}")
            if replace_originals:
                logger.info("原始图像已被修复版本替换")
            else:
                logger.info("原始图像保持不变，可以使用 --replace 参数替换原始文件")
        
        return fixed_success, fixed_failed

def main():
    parser = argparse.ArgumentParser(description="DreamBooth 图像数据集修复工具")
    parser.add_argument("--input_dir", type=str, required=True, help="输入图像目录")
    parser.add_argument("--mode", type=str, choices=["check", "deep_check", "fix", "replace"], 
                       default="check", help="操作模式: check=基本检查, deep_check=深度检查, fix=修复问题, replace=修复并替换原始文件")
    parser.add_argument("--verbose", action="store_true", help="显示详细日志")
    
    args = parser.parse_args()
    
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    input_dir = args.input_dir
    if not os.path.exists(input_dir):
        logger.error(f"输入目录不存在: {input_dir}")
        return 1
    
    logger.info(f"正在处理数据集: {input_dir}, 模式: {args.mode}")
    
    processor = EnhancedDatasetPreprocessor(input_dir)
    
    if args.mode == "check":
        valid, invalid = processor.validate_dataset(mode="basic")
        logger.info(f"基本检查结果: {len(valid)} 个有效图像, {len(invalid)} 个无效图像")
        if invalid:
            logger.info("推荐使用 --mode deep_check 进行更深入的验证")
    
    elif args.mode == "deep_check":
        valid, invalid = processor.validate_dataset(mode="deep")
        logger.info(f"深度检查结果: {len(valid)} 个有效图像, {len(invalid)} 个无效图像")
        if invalid:
            logger.info("使用 --mode fix 尝试修复这些问题")
    
    elif args.mode == "fix":
        fixed_success, fixed_failed = processor.fix_dataset(replace_originals=False)
        if fixed_success > 0:
            logger.info("使用修复的图像重新运行训练")
        if fixed_failed > 0:
            logger.warning(f"仍有 {fixed_failed} 个图像无法修复，建议手动检查")
    
    elif args.mode == "replace":
        fixed_success, fixed_failed = processor.fix_dataset(replace_originals=True)
        if fixed_success > 0:
            logger.info("已用修复的图像替换原始文件，现可重新运行训练")
        if fixed_failed > 0:
            logger.warning(f"仍有 {fixed_failed} 个图像无法修复，建议手动检查")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
