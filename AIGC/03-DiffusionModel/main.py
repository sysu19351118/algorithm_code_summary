import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision.utils import make_grid
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
from denoise_unet import UNet
from ddpm import GaussianDiffusion
import torch
from torchvision.utils import make_grid
import matplotlib.pyplot as plt

import cv2
import numpy as np
import torch

def save_image_grid(tensor, filename, nrow=4, padding=2, normalize=False):
    """
    将批量的图像张量保存为网格图片
    
    参数:
        tensor (torch.Tensor): 形状为 (B, C, H, W) 的图像张量
        filename (str): 保存路径
        nrow (int): 每行显示的图像数量
        padding (int): 图像之间的间距
        normalize (bool): 是否将图像归一化到 [0, 1] 范围
    """
    # 将张量转换为numpy数组并调整通道顺序
    tensor = tensor.detach().cpu()
    
    # 转换为HWC格式的numpy数组
    images = tensor.permute(0, 2, 3, 1).numpy()
    images = (images * 255).astype(np.uint8)  # 转换为0-255范围
    
    # 计算网格的行列数
    batch_size, H, W, C = images.shape
    ncol = int(np.ceil(batch_size / nrow))
    
    # 创建空白网格图像
    grid_h = H * ncol + padding * (ncol + 1)
    grid_w = W * nrow + padding * (nrow + 1)
    grid = np.zeros((grid_h, grid_w, C), dtype=np.uint8)
    grid.fill(255)  # 用白色填充背景
    
    # 将图像填充到网格中
    for i, img in enumerate(images):
        row = i // nrow
        col = i % nrow
        y_start = row * (H + padding) + padding
        y_end = y_start + H
        x_start = col * (W + padding) + padding
        x_end = x_start + W
        grid[y_start:y_end, x_start:x_end] = img
    
    # 保存图像
    if C == 1:  # 灰度图
        grid = grid.squeeze(-1)
    cv2.imwrite(filename, grid)  # BGR转RGB


def train():
    # 参数设置
    device = "cuda" if torch.cuda.is_available() else "cpu"
    image_size = 28
    batch_size = 128
    epochs = 100
    timestep = 1000
    
    # 数据加载
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.Grayscale(num_output_channels=1),  # 转为单通道灰度图
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # 注意这里只有一个值，因为灰度图是单通道
    ])
    
    dataset = datasets.MNIST("/mnt/data2/zzixuantang/algorithm_code_summary/AIGC/03-DiffusionModel/data", train=True, download=True, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    
    # 初始化模型和扩散过程
    model = UNet().to(device)
    diffusion = GaussianDiffusion(timestep=timestep)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    # 添加学习率调度器 - CosineAnnealingLR
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=len(dataloader)*epochs,  # 总迭代次数
        eta_min=1e-6  # 最小学习率
    )
    # 或者可以使用ReduceLROnPlateau
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    #     optimizer, 
    #     mode='min', 
    #     factor=0.5, 
    #     patience=5, 
    #     verbose=True
    # )
    
    # 训练循环
    for epoch in range(epochs):
        pbar = tqdm(dataloader)
        for step, (images, _) in enumerate(pbar):
            optimizer.zero_grad()
            images = images.to(device)
            # print(images.max(), images.min())
            batch_size = images.shape[0]
            # 随机采样时间步
            t = torch.randint(0, timestep, (batch_size,), device=device).long()
            
            # 计算损失
            loss = diffusion.p_losses(model, images, t)
            loss.backward()
            optimizer.step()
            scheduler.step()  # 更新学习率 - 对于CosineAnnealingLR
            
            # 如果使用ReduceLROnPlateau，改为:
            # scheduler.step(loss)
            
            current_lr = optimizer.param_groups[0]['lr']
            pbar.set_description(f"Epoch {epoch} | Loss: {loss.item():.4f} | LR: {current_lr:.2e}")
        
        # 每个epoch结束后采样一些图片
        if epoch % 3 == 0 or epoch == epochs - 1:
            sampled_images = diffusion.sample(model, image_size=image_size, batch_size=16)
            normalized = sampled_images.clone()
            for i in range(len(normalized)):
                normalized[i] -= torch.min(normalized[i])
                normalized[i] *= 1 / torch.max(normalized[i])
            sampled_images = normalized
            
            # 确保值在 [0,1] 范围内（防止数值误差）
            # 保存图片
            save_path = f"/mnt/data2/zzixuantang/algorithm_code_summary/AIGC/03-DiffusionModel/visual/{epoch}.png"
            save_image_grid(sampled_images, save_path)
    # 保存模型
    torch.save(model.state_dict(), "ddpm_model.pth")

if __name__ == "__main__":
    train()