# 导入必要的库
from typing import Any
import os  # 操作系统接口
# 添加 HuggingFace 国内镜像，解决数据集下载问题
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import torch  # PyTorch 核心库
import torch.nn.functional as F  # PyTorch 函数库
import matplotlib.pyplot as plt  # 用于绘制和保存图片
from torchvision import transforms  # 图像预处理工具
from diffusers import DDPMPipeline, DDPMScheduler, UNet2DModel  # 扩散模型相关库
from diffusers.optimization import get_cosine_schedule_with_warmup  # 余弦学习率调度器
from torch.utils.data import DataLoader  # 数据加载器
from datasets import load_dataset  # 加载数据集
import numpy as np  # 数值计算库
from PIL import Image  # 图像处理库


def get_dataloader(dataset_name='mnist', batch_size=128, image_size=32):
    """
    获取数据加载器
    参数:
        dataset_name: 数据集名称
        batch_size: 批次大小
        image_size: 图像尺寸
    返回:
        DataLoader: 数据加载器
    """
    # 定义图像预处理管道
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),  # 调整图像大小
        transforms.ToTensor(),  # 转换为张量
        transforms.Lambda(lambda x: (x * 2) - 1)  # 归一化到[-1, 1]范围
    ])
    
    # 加载MNIST数据集
    dataset = load_dataset("mnist", split="train")
   
    # 预处理函数
    def preprocess(examples):
        # 处理每张图像，确保是灰度图
        images = [transform(image.convert("L") if image.mode != "RGB" else image) 
                 for image in examples["image"]]
        return {"pixel_values": images}
    
    # 设置数据集的预处理函数
    dataset.set_transform(preprocess)
    
    # 自定义数据批处理函数
    def collate_fn(examples):
        # 堆叠图像张量
        images = torch.stack([example["pixel_values"] for example in examples])
        # 转换为连续内存格式并转换为float类型
        images = images.to(memory_format=torch.contiguous_format).float()
        return {"images": images}
    
    # 创建并返回数据加载器
    return DataLoader(
        dataset, 
        batch_size=batch_size,  # 批次大小
        shuffle=True,  # 打乱数据
        collate_fn=collate_fn,  # 批处理函数
        num_workers=0  # 禁用多进程，避免Windows环境下的问题
    )


def setup_model_and_scheduler(image_size=32, in_channels=1, device="cuda"):
    """
    初始化UNet模型和噪声调度器
    参数:
        image_size: 图像尺寸
        in_channels: 输入通道数
        device: 运行设备
    返回:
        model: UNet模型
        noise_scheduler: 噪声调度器
    """
    # 初始化UNet模型
    model = UNet2DModel(
        sample_size=image_size,           # 图片尺寸
        in_channels=in_channels,          # 输入通道数 (1 for MNIST, 3 for CIFAR)
        out_channels=in_channels,         # 输出通道数
        layers_per_block=2,               # 每个残差块的层数
        block_out_channels=(128, 256, 512, 512),  # 各层的通道数配置
        down_block_types=(                # 下采样块类型
            "DownBlock2D",        # 普通下采样块
            "DownBlock2D", 
            "AttnDownBlock2D",    # 带注意力机制的下采样块
            "AttnDownBlock2D",
        ),
        up_block_types=(                  # 上采样块类型
            "AttnUpBlock2D",      # 带注意力机制的上采样块
            "AttnUpBlock2D",
            "UpBlock2D", 
            "UpBlock2D", 
        ),
    )
    # 将模型移到指定设备
    model.to(device)
    
    # 初始化噪声调度器
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=1000,  # 总时间步数
        beta_start=0.0001,         # beta起始值
        beta_end=0.02,             # beta结束值
        beta_schedule="linear",    # 调度策略
        variance_type="fixed_small",  # 方差类型
        prediction_type="epsilon"  # 预测目标：噪声
    )
    
    return model, noise_scheduler


def train_one_epoch(model, noise_scheduler, optimizer, dataloader, device="cuda"):
    """
    训练一个epoch
    参数:
        model: UNet模型
        noise_scheduler: 噪声调度器
        optimizer: 优化器
        dataloader: 数据加载器
        device: 运行设备
    返回:
        float: 平均损失
    """
    # 设置模型为训练模式
    model.train()
    total_loss = 0  # 总损失
    
    # 遍历数据加载器
    for step, batch in enumerate(dataloader):
        # 获取干净图像并移到指定设备
        clean_images = batch["images"].to(device)
        batch_size = clean_images.shape[0]  # 批次大小
        
        # 1. 随机采样时间步
        timesteps = torch.randint(
            0, noise_scheduler.config.num_train_timesteps, 
            (batch_size,), device=device
        ).long()  # 转换为长整型
        
        # 2. 添加噪声 (Diffusers库已封装)
        noise = torch.randn_like(clean_images)  # 生成与图像形状相同的噪声
        noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)  # 添加噪声
        
        # 3. 预测噪声
        noise_pred = model(noisy_images, timesteps, return_dict=False)[0]  # 预测噪声
        
        # 4. 计算损失 - MSE损失
        loss = F.mse_loss(noise_pred, noise)  # 预测噪声与真实噪声的均方误差
        
        # 5. 反向传播
        optimizer.zero_grad()  # 清零梯度
        loss.backward()  # 计算梯度
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # 梯度裁剪，防止梯度爆炸
        optimizer.step()  # 更新参数
        
        total_loss += loss.item()  # 累加损失
        
        # 每100步打印进度
        if step % 100 == 0:
            print(f"Step {step}: Loss = {loss.item():.4f}")
    
    # 返回平均损失
    return total_loss / len(dataloader)


@torch.no_grad()  # 禁用梯度计算，节省内存

def generate_images(model, noise_scheduler, num_images=16, image_size=32, in_channels=1, device="cuda"):
    """
    生成新图片
    参数:
        model: UNet模型
        noise_scheduler: 噪声调度器
        num_images: 生成图片数量
        image_size: 图像尺寸
        in_channels: 输入通道数
        device: 运行设备
    返回:
        numpy.array: 生成的图片数组
    """
    # 设置模型为评估模式
    model.eval()
    
    # 1. 创建pipeline
    pipeline = DDPMPipeline(
        unet=model,
        scheduler=noise_scheduler
    )
    # 将pipeline移到指定设备
    pipeline.to(device)
    
    # 2. 生成图片
    images = pipeline(
        batch_size=num_images,  # 批量生成数量
        generator=torch.Generator(device=device).manual_seed(42)  # 固定随机种子，保证可重复性
    ).images
    
    # 3. 转换格式: PIL Image -> numpy array
    images_np = np.stack([np.array(img) for img in images])
    
    return images_np


def save_image_grid(images, filename="generated_grid.png", ncols=4):
    """
    保存图片网格
    参数:
        images: 图片数组
        filename: 保存文件名
        ncols: 列数
    """
    # 计算行数
    nrows = len(images) // ncols
    # 创建画布和子图
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols*2, nrows*2))
    # 将二维数组展平为一维
    axes = axes.flatten()
    
    # 遍历每张图片
    for idx, img in enumerate(images):
        # 显示图片，灰度图使用gray颜色映射
        axes[idx].imshow(img, cmap='gray' if img.ndim == 2 else None)
        # 关闭坐标轴
        axes[idx].axis('off')
    
    # 调整布局
    plt.tight_layout()
    # 保存图片
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    # 显示图片
    plt.show()
    # 打印保存路径
    print(f"已保存到: {filename}")


def main():
    # 配置
    device = "cuda" if torch.cuda.is_available() else "cpu"  # 自动选择设备
    print(f"使用设备: {device}")
    
    # 1. 准备数据
    print("加载数据集...")
    dataloader = get_dataloader(dataset_name='mnist', batch_size=32, image_size=32)  # 减小batch_size
    
    # 2. 初始化模型和调度器
    print("初始化模型...")
    model, noise_scheduler = setup_model_and_scheduler(
        image_size=32, 
        in_channels=1,  # MNIST是灰度图
        device=device
    )
    
    # 3. 初始化优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)  # AdamW优化器
    
    # 4. 训练循环
    num_epochs = 10  # 训练轮数（减小以加快训练）
    print(f"开始训练，共{num_epochs}个epoch...")
    
    for epoch in range(num_epochs):
        # 训练一个epoch
        avg_loss = train_one_epoch(model, noise_scheduler, optimizer, dataloader, device)
        
        # 打印进度
        print(f"Epoch {epoch+1}/{num_epochs} | Loss: {avg_loss:.4f}")
        
        # 每2个epoch生成并保存样本
        if (epoch + 1) % 2 == 0 or epoch == 0:
            print(f"生成样本...")
            images = generate_images(
                model, noise_scheduler, 
                num_images=16, 
                image_size=32, 
                in_channels=1,
                device=device
            )
            
            # 保存模型
            model_dir = "saved_models"  # 模型保存目录
            os.makedirs(model_dir, exist_ok=True)  # 创建目录（如果不存在）
            torch.save(model.state_dict(), f"{model_dir}/model_epoch_{epoch+1}.pth")  # 保存模型权重
            print(f"模型已保存到: {model_dir}/model_epoch_{epoch+1}.pth")
            
            # 保存生成的图片
            save_image_grid(images, f"generated_epoch_{epoch+1}.png")
    
    # 5. 最终生成
    print("训练完成，生成最终结果...")
    final_images = generate_images(
        model, noise_scheduler, 
        num_images=16, 
        image_size=32, 
        in_channels=1,
        device=device
    )
    save_image_grid(final_images, "final_generated.png")
    
    return model, noise_scheduler


@torch.no_grad()  # 禁用梯度计算

def create_denoising_animation(model, noise_scheduler, num_steps=50, device="cuda"):
    """
    创建去噪过程的动画
    参数:
        model: UNet模型
        noise_scheduler: 噪声调度器
        num_steps: 动画步数
        device: 运行设备
    返回:
        list: 去噪过程中的图片列表
    """
    # 设置模型为评估模式
    model.eval()
    
    # 1. 生成初始噪声
    batch_size = 1  # 批大小为1
    image_size = 32  # 图像尺寸
    in_channels = 1  # 输入通道数
    
    # 生成初始噪声
    noise = torch.randn(
        (batch_size, in_channels, image_size, image_size),
        device=device
    )
    
    # 2. 存储每一步的结果
    images = []  # 存储去噪过程的图片
    x = noise.clone()  # 初始化为噪声
    
    # 3. 逐步去噪
    # 生成时间步列表，从大到小
    timesteps = list(reversed(range(0, noise_scheduler.config.num_train_timesteps, 
                                   noise_scheduler.config.num_train_timesteps // num_steps)))
    
    # 遍历每个时间步
    for i, t in enumerate(timesteps):
        # 构建时间步张量
        timestep = torch.tensor([t] * batch_size, device=device)
        
        # 预测噪声
        noise_pred = model(x, timestep, return_dict=False)[0]
        
        # 使用调度器计算前一步
        x = noise_scheduler.step(
            noise_pred, t, x, generator=None
        ).prev_sample
        
        # 转换为PIL图片
        if i % 5 == 0 or i == len(timesteps) - 1:  # 每5步保存一次，最后一步也要保存
            img = x.squeeze().cpu().numpy()  # 移除batch维度并移到CPU
            img = np.clip((img + 1) * 127.5, 0, 255).astype(np.uint8)  # 转换为0-255范围
            images.append(Image.fromarray(img))  # 添加到列表
        
        # 打印进度
        print(f"去噪进度: {i+1}/{len(timesteps)}")
    
    # 4. 保存为GIF
    images[0].save(
        "denoising_process.gif",  # 保存文件名
        save_all=True,  # 保存所有帧
        append_images=images[1:],  # 后续帧
        duration=100,  # 每帧100ms
        loop=0  # 无限循环
    )
    print("去噪动画已保存为: denoising_process.gif")
    
    return images


if __name__ == "__main__":
    # 运行主函数
    model, scheduler = main()
    
    # 使用训练好的模型创建动画
    # 注意：这里需要先定义device变量
    device = "cuda" if torch.cuda.is_available() else "cpu"
    create_denoising_animation(model, scheduler, num_steps=100, device=device)
