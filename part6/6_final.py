"""
Vision Transformer (ViT) 图像分类与注意力可视化

本脚本实现以下功能：
1. Linear Probing: 冻结预训练模型参数，只训练分类头
2. Full Fine-tuning: 解冻所有参数进行微调
3. 注意力可视化: 提取并可视化 ViT 的注意力权重

代码结构：
- 第一部分：导入库和环境配置
- 第二部分：模型加载和数据准备
- 第三部分：Linear Probing 训练
- 第四部分：Full Fine-tuning 训练
- 第五部分：注意力权重提取与可视化
"""

# ==================== 第一部分：导入库和环境配置 ====================

import os
# 设置 HuggingFace 镜像，加速模型下载
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from timm import create_model
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端，避免 GUI 问题
import matplotlib.pyplot as plt
import numpy as np
import cv2
from PIL import Image

# ==================== 第二部分：模型加载和数据准备 ====================

# 设置设备：优先使用 GPU，没有 GPU 则使用 CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# 加载预训练的 ViT-Base 模型
# vit_base_patch16_224 表示：
#   - base: 模型规模（还有 small, large, huge）
#   - patch16: 每个 patch 大小为 16x16 像素
#   - 224: 输入图像大小为 224x224
model = create_model('vit_base_patch16_224', pretrained=True)

# 修改分类头：从 1000 类（ImageNet）改为 10 类（CIFAR-10）
# ViT 的分类头是一个线性层，输入维度是 768（base 模型的隐藏维度）
features = model.head.in_features  # 获取输入特征维度：768
model.head = nn.Linear(features, 10)  # 创建新的分类头：768 -> 10

# 将模型移动到指定设备（GPU 或 CPU）
model = model.to(device)

# 图像预处理流水线
# ViT 需要特定的预处理：Resize -> ToTensor -> Normalize
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # ViT 输入大小固定为 224x224
    transforms.ToTensor(),  # 转换为张量，像素值从 [0,255] 变为 [0,1]
    transforms.Normalize(  # 使用 ImageNet 的均值和标准差进行归一化
        mean=[0.485, 0.456, 0.406],  # ImageNet 的 RGB 均值
        std=[0.229, 0.224, 0.225]    # ImageNet 的 RGB 标准差
    )
])

# 加载 CIFAR-10 数据集
# CIFAR-10 包含 10 个类别，每张图片大小为 32x32
train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

# 创建数据加载器
# batch_size=32: 每次迭代处理 32 张图片
# shuffle=True: 训练集打乱顺序，测试集不需要打乱
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# ==================== 训练和评估函数 ====================

def train_one_epoch(model, loader, criterion, optimizer, device):
    """
    训练一个 epoch
    
    参数：
        model: 神经网络模型
        loader: 数据加载器
        criterion: 损失函数
        optimizer: 优化器
        device: 计算设备
    
    返回：
        avg_loss: 平均损失
        accuracy: 准确率
    """
    model.train()  # 设置为训练模式（启用 Dropout、BatchNorm 等）
    total_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels in loader:
        # 将数据移动到指定设备
        images, labels = images.to(device), labels.to(device)
        
        # 前向传播
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # 反向传播
        optimizer.zero_grad()  # 清零梯度
        loss.backward()        # 计算梯度
        optimizer.step()       # 更新参数
        
        # 统计损失和准确率
        total_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    return total_loss / total, correct / total

def evaluate(model, loader, device):
    """
    评估模型在测试集上的准确率
    
    参数：
        model: 神经网络模型
        loader: 数据加载器
        device: 计算设备
    
    返回：
        accuracy: 准确率
    """
    model.eval()  # 设置为评估模式（禁用 Dropout、BatchNorm 等）
    correct = 0
    total = 0
    
    with torch.no_grad():  # 禁用梯度计算，节省内存和加速
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    return correct / total

# ==================== 第三部分：Linear Probing ====================
"""
Linear Probing（线性探测）：
- 冻结预训练模型的所有参数
- 只训练最后的分类头（线性层）
- 优点：训练速度快，参数少，适合快速验证模型效果
- 缺点：可能无法充分发挥预训练模型的潜力
"""

print("\n=== Linear Probing ===")

# 冻结所有参数
for param in model.parameters():
    param.requires_grad = False

# 解冻分类头参数
for param in model.head.parameters():
    param.requires_grad = True

# 设置损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.head.parameters(), lr=1e-5)  # 只优化分类头参数

# 训练
num_epochs = 3
lp_test_accs = []

for epoch in range(num_epochs):
    train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
    test_acc = evaluate(model, test_loader, device)
    lp_test_accs.append(test_acc)
    print(f"Epoch {epoch+1}/{num_epochs}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}")

# ==================== 第四部分：Full Fine-tuning ====================
"""
Full Fine-tuning（全参数微调）：
- 解冻模型的所有参数
- 使用较小的学习率进行训练
- 优点：可以充分发挥预训练模型的潜力
- 缺点：训练时间长，需要更多内存，可能过拟合
"""

print("\n=== Full Fine-tuning ===")

# 解冻所有参数
for param in model.parameters():
    param.requires_grad = True

# 使用更小的学习率，避免破坏预训练权重
optimizer = optim.Adam(model.parameters(), lr=1e-7)

# 训练
ft_test_accs = []

for epoch in range(num_epochs):
    train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
    test_acc = evaluate(model, test_loader, device)
    ft_test_accs.append(test_acc)
    print(f"Epoch {epoch+1}/{num_epochs}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}")

# 绘制准确率对比图
plt.figure(figsize=(10, 5))
plt.plot(range(1, num_epochs+1), lp_test_accs, label='Linear Probing Test Acc')
plt.plot(range(1, num_epochs+1), ft_test_accs, label='Full Fine-tuning Test Acc')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Accuracy Comparison: Linear Probing vs Full Fine-tuning')
plt.legend()
plt.savefig('accuracy_comparison.png')
print("\nAccuracy comparison saved to accuracy_comparison.png")

# ==================== 第五部分：注意力权重提取与可视化 ====================
"""
注意力可视化原理：
1. ViT 将图像分割成多个 patch（16x16 像素）
2. 每个 patch 被视为一个 token
3. Transformer 使用自注意力机制让每个 token 关注其他 token
4. 我们提取 CLS token 对其他 patch 的注意力权重
5. 将注意力权重可视化为热力图，显示模型关注图像的哪些区域

关键技术点：
- ViT 的注意力层输出的是加权后的特征，不是注意力权重本身
- 我们需要通过 Hook 函数在注意力层计算时手动提取注意力权重
- 注意力权重的形状是 [batch, num_heads, seq_len, seq_len]
"""

print("\n=== Generating Attention Heatmaps ===")

# 全局变量，用于存储注意力权重
attention_weights = None

def attention_hook(module, input, output):
    """
    Hook 函数：在注意力层前向传播时被调用
    
    参数：
        module: 当前层（注意力层）
        input: 输入张量
        output: 输出张量
    
    注意：
        - ViT 的注意力层输出是加权后的特征，不是注意力权重
        - 我们需要从输入重新计算注意力权重
    """
    global attention_weights
    
    # 获取输入特征
    # input[0] 的形状: [batch_size, seq_len, hidden_dim]
    # seq_len = 196 (14x14 patches) + 1 (CLS token) = 197
    x = input[0]
    B, N, C = x.shape  # Batch, Sequence Length, Channels
    
    # 计算 Q、K、V
    # module.qkv 是一个线性层，将输入投影到 Q、K、V 三个空间
    # 输出形状: [batch, seq_len, 3 * hidden_dim]
    qkv = module.qkv(x)
    
    # 重塑并分离 Q、K、V
    # reshape: [batch, seq_len, 3, num_heads, head_dim]
    # permute: [3, batch, num_heads, seq_len, head_dim]
    qkv = qkv.reshape(B, N, 3, module.num_heads, C // module.num_heads).permute(2, 0, 3, 1, 4)
    q, k, v = qkv[0], qkv[1], qkv[2]  # 每个形状: [batch, num_heads, seq_len, head_dim]
    
    # 计算注意力分数
    # Attention(Q, K) = softmax(Q * K^T / sqrt(d_k))
    # q @ k.transpose(-2, -1): [batch, num_heads, seq_len, seq_len]
    # module.scale = 1 / sqrt(head_dim)，用于缩放
    attn = (q @ k.transpose(-2, -1)) * module.scale
    attn = attn.softmax(dim=-1)  # 对最后一个维度进行 softmax
    
    # 保存注意力权重到全局变量
    attention_weights = attn.detach().cpu()

# 在最后一个 Transformer Block 的注意力层注册 Hook
# 这样每次前向传播时，attention_hook 函数都会被调用
last_block = model.blocks[-1]
last_attn_layer = last_block.attn
last_attn_layer.register_forward_hook(attention_hook)

# 要处理的图片列表
image_files = ['cat1.jpg', 'cat2.jpg', 'dog1.jpg']

for image_file in image_files:
    print(f"\nProcessing {image_file}...")
    
    # 加载图片
    image = Image.open(image_file).convert('RGB')
    
    # 保存原始图片用于显示（resize 到 224x224）
    image_display = np.array(image.resize((224, 224)))
    
    # 预处理图片
    image_tensor = transform(image)
    
    # 添加 batch 维度并移动到设备
    # 形状变化: [3, 224, 224] -> [1, 3, 224, 224]
    input_tensor = image_tensor.unsqueeze(0).to(device)
    
    # 前向传播（此时 Hook 函数会被调用，提取注意力权重）
    with torch.no_grad():
        _ = model(input_tensor)
    
    # 处理注意力权重
    # attention_weights 形状: [batch, num_heads, seq_len, seq_len]
    # seq_len = 197 (196 patches + 1 CLS token)
    
    # 提取 CLS token 对所有 patch 的注意力
    # attention_weights[0, :, 0, 1:] 解释：
    #   [0]: 第一个样本（batch 中只有一个图片）
    #   [:]: 所有注意力头
    #   [0]: CLS token 的位置（第 0 个 token）
    #   [1:]: 对所有 patch 的注意力（跳过 CLS token 自己）
    cls_attn = attention_weights[0, :, 0, 1:].mean(dim=0)  # 对所有头取平均
    
    # 重塑为 14x14 的网格
    # ViT-Base/16 将 224x224 图像分成 14x14 个 patch
    attn_map = cls_attn.reshape(14, 14).numpy()
    
    # 上采样到 224x224
    attn_map = cv2.resize(attn_map, (224, 224))
    
    # 归一化到 [0, 1]
    attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min() + 1e-8)
    
    # 生成热力图
    # COLORMAP_JET: 蓝色（低）-> 绿色 -> 黄色 -> 红色（高）
    heatmap = cv2.applyColorMap(np.uint8(255 * attn_map), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    
    # 将热力图叠加到原始图片上
    # addWeighted: dst = alpha * src1 + beta * src2 + gamma
    # 0.6: 原始图片权重
    # 0.4: 热力图权重
    superimposed = cv2.addWeighted(image_display, 0.6, heatmap, 0.4, 0)
    
    # 保存结果
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(image_display)
    plt.title(f'Original Image ({image_file})')
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(superimposed)
    plt.title('Attention Heatmap')
    plt.axis('off')
    
    output_filename = f'attention_heatmap_{image_file.replace(".jpg", "")}.png'
    plt.savefig(output_filename)
    print(f"Saved: {output_filename}")

print("\n=== All Done! ===")
