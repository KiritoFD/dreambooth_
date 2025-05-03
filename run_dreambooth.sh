#!/bin/bash

# 创建目录
mkdir -p instance_images
mkdir -p output

# 训练命令示例
echo "训练DreamBooth模型示例命令："
echo "python dreambooth_implementation.py --train --instance_data_dir ./instance_images --class_prompt \"a dog\" --model_path ./output"

# 推理命令示例
echo "使用训练好的模型进行推理示例命令："
echo "python dreambooth_implementation.py --infer --model_path ./output --prompt \"a [identifier] dog in a park\" --class_prompt \"a dog\""

# 提示用户
echo ""
echo "使用说明:"
echo "1. 请将您的实例图像放在instance_images目录中"
echo "2. 修改class_prompt以匹配您的主题类别（如：a dog, a cat, a watch等）"
echo "3. 运行训练命令，系统会自动生成稀有标记作为标识符"
echo "4. 训练完成后，使用推理命令生成图像，记得将[identifier]替换为训练时打印的稀有标记"
