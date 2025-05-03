"""
DreamBooth论文关键点理解笔记
论文标题: "DreamBooth: Fine Tuning Text-to-Image Diffusion Models for Subject-Driven Generation"
作者: Nataniel Ruiz, Yuanzhen Li, Varun Jampani, Yael Pritch, Michael Rubinstein, Kfir Aberman
发表于: CVPR 2023
"""

class DreamBoothTheory:
    """DreamBooth的理论基础和关键概念"""
    
    @staticmethod
    def core_problem():
        """核心问题：个性化文本到图像生成"""
        return """
DreamBooth解决的核心问题：
- 如何使已训练的文本到图像模型能够"记住"特定主体的外观
- 只需要少量图像（3-5张）就能学会生成该特定主体
- 同时保持模型对该主体所属类别的一般知识
- 能够在多种场景、姿势和风格中重现该主体

论文引用："While the synthesis capabilities of these models are unprecedented, they lack the ability to mimic the appearance of subjects in a given reference set, and synthesize novel renditions of the same subjects in different contexts."

现有文本到图像模型的局限：
1. 无法准确重现特定主体的外观细节
2. 即使使用详尽的文本描述，也难以准确捕捉特定主体的外观
3. 无法保证在不同场景中保持特定主体的一致性
"""
    
    @staticmethod
    def technical_approach():
        """技术方法概述"""
        return """
DreamBooth的核心技术方法：

1. 个性化微调：
   - 对预训练的扩散模型进行微调而非从头训练
   - 只需使用3-5张包含目标主体的图像
   - 保留大部分预训练权重，只更新特定部分

2. 标识符绑定机制：
   - 将特定主体与一个罕见词（如"sks"）绑定
   - 这种绑定使模型能将标识符与特定主体的外观联系起来
   - 形成"a [identifier] [class]"的提示词模式

3. 先验保留损失：
   - 设计特殊的损失函数防止过度拟合和"语言漂移"
   - 使用自动生成的类别图像保留类别一般知识
   - 平衡特定主体的保真度和类别知识的保留

4. 推理技术：
   - 使用"a [identifier] [class] [context]"格式构建提示词
   - 可以指定场景、风格、装扮等元素
   - 模型能够在保持主体特征的同时，根据提示词生成新图像
"""
    
    @staticmethod
    def diffusion_model_background():
        """扩散模型的基础知识"""
        return """
扩散模型的基本原理：

扩散模型是一种概率生成模型，通过将高斯噪声逐步"去噪"来学习数据分布。在论文补充材料中解释：

1. 扩散过程：
   - 前向过程：逐步向原始图像添加噪声，直到完全变为高斯噪声
   - 反向过程：学习如何从噪声中恢复原始图像

2. 条件扩散模型的训练目标：
   Ex,c,,t[wt‖ˆxθ(αtx+σt,c)−x‖²]
   其中：
   - x 是原始图像
   - c 是条件向量（例如从文本提示词获得）
   - ε 是噪声项
   - αt, σt, wt 控制噪声调度
   - t 是扩散时间步

3. 文本到图像扩散模型的实现：
   - 级联模型：使用多个模型逐步提高分辨率 (如Imagen[61])
   - 潜在扩散模型：在低维潜在空间中进行扩散过程 (如Stable Diffusion[59])

4. 文本编码：
   - 使用预训练语言模型（如T5-XXL[53]）将文本转换为条件嵌入
   - 文本首先通过分词器（如SentencePiece）转换为token ID
   - 然后由语言模型生成嵌入向量
"""

    @staticmethod
    def rare_token_binding():
        """稀有标识符绑定的原理"""
        return """
稀有标识符绑定（论文3.2节）：

关键思想：将特定主体与一个罕见词语绑定，形成唯一关联。

1. 标识符选择原则：
   - 从词表中的罕见词（通常是位置5000-10000之间）选择
   - 确保长度短（不超过3个字符）
   - 避免包含空格
   - 例如："sks"、"xxy"、"qrs"等

2. 为什么选择罕见词？
   - 罕见词在预训练数据中几乎不出现，因此没有现有的语义关联
   - 避免与模型已学习的概念冲突
   - 作为"空白容器"，专门用于绑定新的主体特征
   - 防止"语言漂移"：如果使用常见词，模型可能会融合该词的现有知识

3. 提示词结构设计：
   - 训练时使用："a [identifier] [class]"（如"a sks dog"）
   - 推理时使用："a [identifier] [class] [context]"（如"a sks dog on the beach"）
   - 这种结构使模型能够保持特定主体的特征，同时适应不同场景和风格

4. 绑定过程：
   - 通过少量（3-5张）特定主体图像和特定提示词进行微调
   - 模型学习将标识符与特定主体的视觉特征关联起来
   - 一旦训练完成，标识符就能在任何提示词中唤起该特定主体
"""
    
    @staticmethod
    def prior_preservation_loss():
        """先验保留损失的原理与设计"""
        return """
先验保留损失（论文3.3节）：

关键思想：在学习特定主体的同时，保留模型对主体所属类别的一般知识。

1. 问题背景：
   - "语言漂移"：当只使用少量特定主体图像训练时，类别词（如"dog"）可能被重新定义为特定主体
   - 过度拟合：模型可能会丢失对类别的一般理解，导致生成的特定主体缺乏多样性和自然变化

2. 自动生成的类特定先验（Autogenous class-specific prior）：
   - 使用当前模型生成类别图像（如通过"a dog"提示词）
   - 生成约200-300张类别图像
   - 这些图像代表模型当前对该类别的理解

3. 损失函数设计（论文公式1）：
   L = L_instance + λ·L_prior
   
   - L_instance：特定主体损失，确保模型学习特定主体的外观
     使用提示词："a [identifier] [class]"
   
   - L_prior：先验保留损失，确保模型保留类别知识
     使用提示词："a [class]"
   
   - λ：平衡因子，控制两种损失的相对重要性，论文建议值为1.0
     较大的λ会更好地保留类别知识，但可能降低特定主体的保真度
     较小的λ会提高特定主体的保真度，但可能导致类别知识丢失

4. 效果验证：
   - 论文通过消融实验证明，没有先验保留损失时，模型容易过度拟合和出现语言漂移
   - 先验保留损失有效地保持了类别知识，同时实现了高质量的特定主体生成
"""

    @staticmethod
    def training_details():
        """训练细节与实现技巧"""
        return """
训练策略与实现细节（论文3.4节）：

1. 模型组件微调：
   - 联合训练U-Net和文本编码器效果最佳
   - 也可以仅微调U-Net，效果略差但仍然可用（适用于内存受限的情况）
   - 固定VAE编码器和解码器不进行训练

2. 优化参数：
   - 优化器：Adam with β1=0.9, β2=0.999
   - 学习率：5e-6（论文发现这是最佳值）
   - 训练步数：通常500-1000步足够
   - 批量大小：根据GPU内存调整，通常较小（1-4）

3. 数据需求与处理：
   - 特定主体：3-5张高质量图像，不同角度和光照条件效果更好
   - 类别图像：200-300张用于先验保留（通过模型生成）
   - 图像预处理：居中裁剪和调整大小到模型输入尺寸（通常为512x512）

4. 训练流程：
   a. 使用特定主体图像和"a [identifier] [class]"提示词训练
   b. 同时使用生成的类别图像和"a [class]"提示词训练
   c. 两种图像类型混合在一个批次中进行训练
   d. 分别计算两种损失，然后按权重λ组合

5. 内存优化技术：
   - 混合精度训练（FP16）减少内存使用
   - 梯度累积允许使用较小批次大小
   - 可选择性地冻结文本编码器以节省内存
"""

    @staticmethod
    def inference_applications():
        """推理应用场景与效果"""
        return """
应用场景与推理效果（论文图1-5）：

DreamBooth训练完成后，可实现多种创意应用：

1. 主体重新上下文化（Subject Recontextualization）：
   - 将特定主体放置在全新的环境中
   - 提示词示例："a sks dog on the beach", "a sks dog in space"
   - 论文图1展示了多种场景下的重新上下文化效果
   - 模型能够生成自然的光照、阴影和与环境的交互

2. 视角合成（View Synthesis）：
   - 生成特定主体的不同视角和姿势
   - 提示词示例："a front view of sks dog", "a side view of sks dog"
   - 即使训练图像中不包含这些视角，也能生成合理的结果
   - 模型利用对类别的先验知识推断合理的形状和结构

3. 装饰与属性编辑（Accessorization & Property Modification）：
   - 为特定主体添加配件或修改属性
   - 提示词示例："a sks dog wearing a hat", "a sks dog made of gold"
   - 能够保持主体的关键特征，同时自然地融入新属性
   - 论文图4展示了多种属性修改的例子

4. 艺术风格渲染（Artistic Rendering）：
   - 将特定主体以不同艺术风格呈现
   - 提示词示例："a painting of sks dog in van gogh style", "a cartoon sks dog"
   - 能够保持主体特征的同时，适应目标艺术风格
   - 论文图3展示了多种风格渲染效果
"""

    @staticmethod
    def evaluation_metrics():
        """评估指标与方法"""
        return """
评估方法与指标（论文第4节和补充材料）：

DreamBooth提出了一套完整的评估框架：

1. 主体保真度（Subject Fidelity）：
   - 评估生成的特定主体与参考图像的相似度
   - 使用DINO特征相似度：基于自监督学习的视觉特征
   - DINO优于CLIP-I作为保真度指标的原因：DINO训练为区分不同图像，能捕捉更细微的特征差异
   - 人类评估：让评价者判断生成图像中的主体是否与参考主体相同

2. 提示忠实度（Prompt Fidelity）：
   - 评估生成图像是否符合提示词中的场景描述
   - 使用CLIP文本-图像相似度分数
   - 计算生成图像与提示词之间的CLIP相似度

3. 多样性评估（Diversity）：
   - 评估在保持主体一致性的同时，模型生成的多样化程度
   - 使用LPIPS分数衡量不同生成样本之间的视觉差异
   - 高LPIPS分数表示更好的多样性

4. 评估数据集：
   - 包含30个主体（21个物体和9个生物/宠物）
   - 每个主体配有多种提示词：重新上下文化、属性修改、装饰等
   - 每个主体和提示词组合生成4张图像，总计3000张评估图像
   - 这种大规模评估确保了结果的稳健性和可靠性
"""

# 按照流程划分的步骤资源，方便在训练过程中导入
TRAINING_STEPS = {
    "initialization": {
        "title": "初始化与设置",
        "description": """
训练前的初始化与设置步骤：

1. 模型加载：
   - 加载预训练的扩散模型（如Stable Diffusion）
   - 分别加载U-Net、文本编码器、VAE组件

2. 标识符选择：
   - 从词表中选择一个罕见词作为标识符
   - 确保标识符长度短且不包含空格

3. 提示词构建：
   - 构建实例提示词："a [identifier] [class]"
   - 构建类别提示词："a [class]"

4. 环境准备：
   - 设置随机种子确保可复现性
   - 配置混合精度训练以优化内存使用
   - 准备加速器以支持多设备训练
""",
        "code_snippet": """
# 设置随机种子
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)

# 初始化加速器
accelerator = Accelerator(gradient_accumulation_steps=gradient_accumulation_steps,
                          mixed_precision="fp16")

# 加载模型组件
tokenizer = CLIPTokenizer.from_pretrained(model_path, subfolder="tokenizer")
text_encoder = CLIPTextModel.from_pretrained(model_path, subfolder="text_encoder")
vae = AutoencoderKL.from_pretrained(model_path, subfolder="vae")
unet = UNet2DConditionModel.from_pretrained(model_path, subfolder="unet")

# 选择稀有标识符
identifier = find_rare_token(tokenizer)

# 构建提示词
instance_prompt = f"a {identifier} {class_name}"
class_prompt = f"a {class_name}"
"""
    },
    
    "prior_image_generation": {
        "title": "类别先验图像生成",
        "description": """
生成类别先验图像：

1. 目的：
   - 生成用于先验保留损失的类别图像
   - 这些图像代表模型对类别的一般理解

2. 过程：
   - 使用预训练模型和类别提示词（如"a dog"）
   - 通常生成200-300张图像
   - 可以批量生成以提高效率

3. 存储：
   - 将生成的图像保存在指定目录
   - 这些图像将与特定主体图像一起用于训练
""",
        "code_snippet": """
# 创建类别图像目录
class_images_dir = os.path.join(output_dir, "class_images")
os.makedirs(class_images_dir, exist_ok=True)

# 加载生成管道
pipeline = StableDiffusionPipeline.from_pretrained(
    model_path, vae=vae, text_encoder=text_encoder, unet=unet
)
pipeline.to(device)

# 生成类别图像
for i in range(0, prior_generation_samples, batch_size):
    batch_prompts = [class_prompt] * min(batch_size, prior_generation_samples - i)
    outputs = pipeline(batch_prompts, num_inference_steps=50, guidance_scale=7.5)
    
    for j, image in enumerate(outputs.images):
        image.save(os.path.join(class_images_dir, f"class_{i+j:04d}.png"))
"""
    },
    
    "dataset_preparation": {
        "title": "数据集准备",
        "description": """
准备训练数据集：

1. 数据组成：
   - 特定主体图像：用户提供的3-5张主体图像
   - 类别先验图像：之前生成的200-300张类别图像

2. 图像处理：
   - 中心裁剪保证图像比例一致
   - 调整大小到模型输入尺寸（通常512x512）
   - 标准化像素值到[-1, 1]区间

3. 标签构建：
   - 为每张图像添加标签：是特定主体还是类别图像
   - 这将决定使用哪种提示词和损失函数
""",
        "code_snippet": """
# 创建数据集
dataset = DreamBoothDataset(
    instance_images_path=instance_data_dir,
    class_images_path=class_images_dir,
    tokenizer=tokenizer,
    size=512
)

# 创建数据加载器
dataloader = DataLoader(dataset, batch_size=train_batch_size, shuffle=True)

# 准备文本嵌入
instance_text_inputs = tokenizer(
    instance_prompt,
    padding="max_length",
    max_length=tokenizer.model_max_length,
    truncation=True,
    return_tensors="pt"
)

class_text_inputs = tokenizer(
    class_prompt,
    padding="max_length",
    max_length=tokenizer.model_max_length,
    truncation=True,
    return_tensors="pt"
)
"""
    },
    
    "optimization_setup": {
        "title": "优化器与训练设置",
        "description": """
配置优化器与训练参数：

1. 优化器设置：
   - 论文推荐使用Adam优化器
   - 学习率设为5e-6
   - Beta参数设为(0.9, 0.999)

2. 训练参数：
   - 根据论文设置训练步数（通常500-1000步）
   - 配置梯度累积步数以支持小批量训练
   - 设置混合精度训练以优化内存使用

3. 模型设置：
   - 决定是否同时训练U-Net和文本编码器
   - 固定VAE不进行训练以保持重构质量
""",
        "code_snippet": """
# 配置待优化参数
if train_text_encoder:
    params_to_optimize = list(unet.parameters()) + list(text_encoder.parameters())
else:
    params_to_optimize = unet.parameters()

# 设置优化器
optimizer = torch.optim.AdamW(
    params_to_optimize,
    lr=learning_rate,
    betas=(0.9, 0.999),
    weight_decay=1e-2
)

# 设置噪声调度器
noise_scheduler = DDPMScheduler.from_pretrained(model_path, subfolder="scheduler")

# 准备加速训练
unet, text_encoder, optimizer, dataloader = accelerator.prepare(
    unet, text_encoder, optimizer, dataloader
)

# VAE设置为评估模式并固定权重
vae.requires_grad_(False)
vae.eval()
vae.to(accelerator.device, dtype=torch.float32)
"""
    },
    
    "training_loop": {
        "title": "训练循环与损失计算",
        "description": """
训练过程的核心循环：

1. 训练步骤：
   - 遍历数据集的每个批次
   - 区分特定主体图像和类别图像
   - 分别计算两种损失并按权重组合

2. 损失计算：
   - 实例损失(L_instance)：使用特定主体图像计算
   - 先验损失(L_prior)：使用类别图像计算
   - 总损失：L = L_instance + λ·L_prior

3. 扩散过程：
   - 给潜在表示添加随机噪声
   - 训练模型预测添加的噪声
   - 这符合扩散模型的标准训练方法
""",
        "code_snippet": """
# 主训练循环
progress_bar = tqdm(range(max_train_steps))
global_step = 0

for epoch in range(num_epochs):
    for step, batch in enumerate(dataloader):
        with accelerator.accumulate(unet):
            # 准备输入
            pixel_values = batch["pixel_values"].to(accelerator.device)
            is_instance = batch["is_instance"]

            # 潜在空间编码
            with torch.no_grad():
                latents = vae.encode(pixel_values).latent_dist.sample()
                latents = latents * vae.config.scaling_factor

            # 添加噪声
            noise = torch.randn_like(latents)
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, 
                                      (latents.shape[0],), device=latents.device)
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            # 准备文本条件
            instance_embeds = text_encoder(instance_text_inputs.input_ids.to(accelerator.device))[0]
            class_embeds = text_encoder(class_text_inputs.input_ids.to(accelerator.device))[0]

            # 计算损失
            instance_loss = 0.0
            class_loss = 0.0

            # 实例(特定主体)损失
            if torch.sum(is_instance).item() > 0:
                instance_pred = unet(
                    noisy_latents[is_instance],
                    timesteps[is_instance],
                    encoder_hidden_states=instance_embeds.repeat(torch.sum(is_instance).item(), 1, 1)
                ).sample
                instance_loss = F.mse_loss(instance_pred, noise[is_instance])

            # 类别(先验保留)损失
            if torch.sum(~is_instance).item() > 0:
                class_pred = unet(
                    noisy_latents[~is_instance],
                    timesteps[~is_instance],
                    encoder_hidden_states=class_embeds.repeat(torch.sum(~is_instance).item(), 1, 1)
                ).sample
                class_loss = F.mse_loss(class_pred, noise[~is_instance])

            # 组合损失
            loss = instance_loss + prior_preservation_weight * class_loss

            # 反向传播
            accelerator.backward(loss)
            optimizer.step()
            optimizer.zero_grad()

        # 更新进度条
        progress_bar.update(1)
        global_step += 1
        
        if global_step >= max_train_steps:
            break
"""
    },
    
    "model_saving": {
        "title": "模型保存与验证",
        "description": """
模型保存和训练结果验证：

1. 保存模型：
   - 保存微调后的U-Net权重
   - 如果训练了文本编码器，也保存其权重
   - 保存标识符以便后续使用

2. 测试验证：
   - 使用训练好的模型生成几个测试图像
   - 验证特定主体在不同场景中的表现
   - 检查主体特征的保留情况

3. 结果展示：
   - 生成多种场景下的特定主体图像
   - 展示不同风格和上下文中的主体表现
""",
        "code_snippet": """
# 等待所有进程完成
accelerator.wait_for_everyone()

# 解包装模型
unet = accelerator.unwrap_model(unet)
if train_text_encoder:
    text_encoder = accelerator.unwrap_model(text_encoder)

# 保存标识符
os.makedirs(output_dir, exist_ok=True)
with open(os.path.join(output_dir, "identifier.txt"), "w") as f:
    f.write(identifier)

# 保存模型
pipeline = StableDiffusionPipeline.from_pretrained(
    model_path,
    unet=unet,
    text_encoder=text_encoder if train_text_encoder else None,
    tokenizer=tokenizer,
    scheduler=noise_scheduler,
    vae=vae
)
pipeline.save_pretrained(output_dir)

# 生成测试图像
test_prompts = [
    f"a {identifier} {class_name} on the beach",
    f"a {identifier} {class_name} in the snow",
    f"a painting of {identifier} {class_name} in van gogh style"
]

for i, prompt in enumerate(test_prompts):
    image = pipeline(prompt).images[0]
    image.save(f"test_{i}.png")
"""
    },
}

def print_theory_step(step_number, title, description):
    """打印训练步骤的理论解释"""
    print("\n" + "="*80)
    print(f"【论文解析 {step_number}】{title}")
    print("="*80)
    print(description)

# 主要训练步骤的理论解释字典
THEORY_STEPS = {
    "initialization": {
        "title": "初始化与标识符选择 (论文3.2节)",
        "description": DreamBoothTheory.rare_token_binding()
    },
    "prior_preservation": {
        "title": "先验保留机制 (论文3.3节)",
        "description": DreamBoothTheory.prior_preservation_loss()
    },
    "loss_function": {
        "title": "损失函数设计 (论文公式1)",
        "description": """
DreamBooth的损失函数包含两部分，在论文公式(1)中定义:
L = L_instance + λ·L_prior

各部分含义:
- L_instance: 确保模型学习特定主体的外观特征，使用标识符提示词
  提示词格式: "a [identifier] [class]" (如"a sks dog")
  
- L_prior: 确保模型保留类别的一般知识，防止过度拟合
  提示词格式: "a [class]" (如"a dog")
  
- λ: 控制两种损失的平衡权重，论文建议值为1.0
  较大的λ会更好地保留类别知识，但可能降低对特定主体的保真度
  较小的λ会提高特定主体的保真度，但可能导致类别知识丢失

两种损失都是使用扩散模型的噪声预测MSE损失计算的，遵循标准扩散训练方法。
"""
    },
    "training": {
        "title": "训练策略优化 (论文3.4节)",
        "description": DreamBoothTheory.training_details()
    },
    "completion": {
        "title": "DreamBooth应用效果 (论文图3-5)",
        "description": DreamBoothTheory.inference_applications()
    }
}

def get_theory_step(step_name):
    """获取指定步骤的理论解释"""
    if step_name in THEORY_STEPS:
        return THEORY_STEPS[step_name]
    return None

def get_training_step(step_name):
    """获取指定训练步骤的详细说明和代码示例"""
    if step_name in TRAINING_STEPS:
        return TRAINING_STEPS[step_name]
    return None

# 论文概述，用于整体了解
def paper_overview():
    """DreamBooth论文整体概述"""
    print("\n" + "="*100)
    print("DreamBooth: Fine Tuning Text-to-Image Diffusion Models for Subject-Driven Generation".center(100))
    print("="*100)
    
    print("\n【研究背景】")
    print(DreamBoothTheory.core_problem())
    
    print("\n【主要贡献】")
    print("""
1. 提出了一种文本到图像扩散模型的个性化微调方法，只需少量图像
2. 设计了标识符绑定机制，将特定主体与罕见词绑定
3. 开发了自动生成的类特定先验保留损失，防止过度拟合和语言漂移
4. 证明能在不同场景、视角和风格中保持主体特征，实现多样化生成
5. 建立了评估主体驱动生成的新数据集和评估协议
""")
    
    print("\n【技术方法概述】")
    print(DreamBoothTheory.technical_approach())
    
    print("\n【应用场景】")
    print("""
DreamBooth可用于多种创意应用场景：
1. 个人物品的虚拟展示和在线营销
2. 宠物在不同场景中的照片生成
3. 个人化的艺术创作和风格化图像
4. 产品设计和原型可视化
5. 虚拟试衣和配饰展示
""")

# 仅当直接运行此文件时才执行概述
if __name__ == "__main__":
    paper_overview()
