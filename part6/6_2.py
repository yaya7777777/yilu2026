import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from timm import create_model
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import cv2
from PIL import Image

# 全局变量，用于存储 Attention 权重
attention_weights = None

# 钩子函数：在最后一层 Transformer Block 的 Attention 层执行后，提取权重
def attention_hook(module, input, output):
    global attention_weights
    # timm 1.0.25 版本的 Attention 输出格式
    try:
        if isinstance(output, tuple):
            # output[0] 是注意力输出，output[1] 是注意力权重
            if len(output) >= 2 and output[1] is not None:
                attention_weights = output[1].detach().cpu()
            else:
                attention_weights = None
        else:
            attention_weights = None
    except Exception as e:
        print(f"钩子函数错误: {e}")
        attention_weights = None

def generate_attention_map(model, image_path, transform, device):
    """为单个图像生成并保存注意力热力图"""
    print(f"正在为 {image_path} 生成注意力图...")
    try:
        image = Image.open(image_path).convert('RGB')
        image = image.resize((224, 224))

        # 预处理,和训练时保持一致
        input_tensor = transform(image).unsqueeze(0).to(device)

        # 前向传播以触发钩子
        with torch.no_grad():
            _ = model(input_tensor)

        # 处理权重：取 CLS Token 对所有 Image Patches 的注意力
        if attention_weights is not None:
            # 取 CLS Token 对所有 Image Patches 的注意力
            cls_attn = attention_weights[0, :, 0, 1:].mean(dim=0)
            # 重塑为 14x14
            attn_map = cls_attn.reshape(14, 14).numpy()
            # 上采样到 224x224
            attn_map = cv2.resize(attn_map, (224, 224))
            # 归一化
            attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min())
            # 生成热力图
            image_np = np.array(image)
            heatmap = cv2.applyColorMap(np.uint8(255 * attn_map), cv2.COLORMAP_JET)
            heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
            # 叠加
            superimposed = cv2.addWeighted(image_np, 0.6, heatmap, 0.4, 0)
            # 保存
            plt.figure(figsize=(12, 6))
            plt.subplot(1, 2, 1)
            plt.imshow(image_np)
            plt.title('Original Image')
            plt.axis('off')
            plt.subplot(1, 2, 2)
            plt.imshow(superimposed)
            plt.title('Attention Heatmap')
            plt.axis('off')
            
            # 从原始路径生成保存路径
            save_path = os.path.splitext(image_path)[0] + '_heatmap.png'
            plt.savefig(save_path)
            print(f"注意力热力图已保存为 {save_path}")
            plt.close() # 关闭图像，防止在循环中重复显示
        else:
            print(f"无法为 {image_path} 获取注意力权重")
    except Exception as e:
        print(f"为 {image_path} 生成热力图时出错: {e}")

def main():
    print("开始加载模型...")
    # 加载预训练 ViT base 模型
    model = create_model('vit_base_patch16_224', pretrained=True)
    print("模型加载成功！")
    # 修改头部：从 1000 类改为 10 类
    features = model.head.in_features
    model.head = nn.Linear(features, 10)

    # --- 步骤1: 迁移学习 ---
    # Linear Probing：冻结除 Head 外所有参数
    for param in model.parameters():
        param.requires_grad = False
    for param in model.head.parameters():
        param.requires_grad = True
        
    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()  # 多分类任务用交叉熵损失
    optimizer = optim.Adam(model.head.parameters(), lr=1e-5)  # 只优化分类头

    # 图像预处理
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 加载 CIFAR-10 训练集和测试集
    print("加载 CIFAR-10 数据集...")
    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    # 数据加载器
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=0)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    num_epochs = 3  # 增加 epoch 数量以获得更好的曲线
    lp_train_accs = []
    lp_test_accs = []

    print("\n=== 开始 Linear Probing ===")
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        batch_count = 0
        total_batches = len(train_loader)
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            batch_count += 1
            
            # 每100个batch打印一次进度
            if (batch_idx + 1) % 100 == 0 or batch_idx == 0:
                print(f"  Epoch {epoch+1}/{num_epochs} - Batch [{batch_idx+1}/{total_batches}] Loss: {loss.item():.4f}")
        
        # 评估
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        test_acc = correct / total
        lp_test_accs.append(test_acc)
        avg_loss = train_loss / batch_count
        print(f"Epoch {epoch+1}/{num_epochs} 完成 - Avg Loss: {avg_loss:.4f}, Test Acc: {test_acc:.4f}")

    # Full Fine-tuning：解冻所有参数
    for param in model.parameters():
        param.requires_grad = True
        
    optimizer = optim.Adam(model.parameters(), lr=1e-5) # 调整学习率

    print("\n=== 开始 Full Fine-tuning ===")
    print("注意: 此阶段训练所有参数，速度较慢，请耐心等待...")
    ft_test_accs = []
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        batch_count = 0
        total_batches = len(train_loader)
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            batch_count += 1
            
            # 每100个batch打印一次进度
            if (batch_idx + 1) % 100 == 0 or batch_idx == 0:
                print(f"  Epoch {epoch+1}/{num_epochs} - Batch [{batch_idx+1}/{total_batches}] Loss: {loss.item():.4f}")

        # 评估
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        test_acc = correct / total
        ft_test_accs.append(test_acc)
        avg_loss = train_loss / batch_count
        print(f"Epoch {epoch+1}/{num_epochs} 完成 - Avg Loss: {avg_loss:.4f}, Test Acc: {test_acc:.4f}")
        
    # 绘制准确率对比曲线
    total_epochs = num_epochs * 2
    all_epochs = list(range(1, total_epochs + 1))
    all_accs = lp_test_accs + ft_test_accs
    
    plt.figure(figsize=(12, 6))
    
    # 绘制连续曲线
    plt.plot(all_epochs, all_accs, 'b-', linewidth=2, alpha=0.7)
    
    # 绘制 Linear Probing 阶段的点
    lp_epochs = list(range(1, num_epochs + 1))
    plt.plot(lp_epochs, lp_test_accs, 'bo-', markersize=8, linewidth=2, 
             label=f'Linear Probing ({num_epochs} epochs)', marker='o')
    
    # 绘制 Full Fine-tuning 阶段的点
    ft_epochs = list(range(num_epochs + 1, total_epochs + 1))
    plt.plot(ft_epochs, ft_test_accs, 'rs-', markersize=8, linewidth=2,
             label=f'Full Fine-tuning ({num_epochs} epochs)', marker='s')
    
    # 添加阶段分隔线
    plt.axvline(x=num_epochs + 0.5, color='gray', linestyle='--', alpha=0.5, 
                label='Stage Transition')
    
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Test Accuracy', fontsize=12)
    plt.title('Linear Probing vs Full Fine-tuning Accuracy Comparison', fontsize=14)
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.xticks(all_epochs)
    plt.ylim([min(all_accs) - 0.05, max(all_accs) + 0.05])
    plt.tight_layout()
    plt.savefig('accuracy_comparison.png', dpi=150, bbox_inches='tight')
    print("\n准确率对比图已保存为 accuracy_comparison.png")
    plt.close()

    # --- 步骤2: Attention Map 可视化 ---
    print("\n=== 开始生成注意力热力图 ===")
    
    # 禁用融合注意力以获取注意力权重
    if hasattr(model.blocks[-1].attn, 'fused_attn'):
        model.blocks[-1].attn.fused_attn = False
    
    # 注册钩子函数
    last_attn_layer = model.blocks[-1].attn
    last_attn_layer.register_forward_hook(attention_hook)
    
    # 获取part6目录下的所有图片
    image_files = [f for f in os.listdir('.') if f.endswith(('.jpg', '.jpeg', '.png'))]
    
    for image_file in image_files:
        generate_attention_map(model, image_file, transform, device)

    print("\n所有任务执行完成！")

if __name__ == '__main__':
    main()
