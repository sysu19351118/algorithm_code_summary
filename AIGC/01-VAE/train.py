import pdb
import pytorch_lightning as pl
from miniset_data_module import MNISTDataModule
from VAE import MNISTVAE
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import torch
import os

import torch
import matplotlib.pyplot as plt
import os
from pytorch_lightning.callbacks import Callback


class VAESampleCallback(Callback):
    def __init__(self, save_path, sample_interval=5, num_samples=4):
        self.sample_interval = sample_interval  # 每隔多少epoch采样一次
        self.num_samples = num_samples          # 每次采样数量
        self.save_path = save_path
        os.makedirs(save_path, exist_ok=True)  # 创建保存目录

    def on_train_epoch_end(self, trainer, pl_module):
        if (trainer.current_epoch + 1) % self.sample_interval == 0:
            # 使用VAE生成样本
            with torch.no_grad():
                pl_module.eval()
                samples = pl_module.sample(self.num_samples)  # 假设你的VAE有sample方法
                pl_module.train()
            samples_ = []
            for simg in samples:
                simg = (simg-simg.min())/(simg.max()-simg.min())
                samples_.append(simg)
            # 将生成的样本保存为一张图片
            fig, axes = plt.subplots(1, self.num_samples, figsize=(self.num_samples*2, 2))
            for i, ax in enumerate(axes):
                ax.imshow(samples_[i].cpu().numpy().reshape(28, 28), cmap='gray')
                ax.axis('off')
            
            plt.tight_layout()
            plt.savefig(f"{self.save_path}/epoch_{trainer.current_epoch:03d}.png")
            plt.close()
# 定义模型保存
import os
import torch
import matplotlib.pyplot as plt
from pytorch_lightning.callbacks import Callback

class VAEReconstructionCallback(Callback):
    def __init__(self, save_path, num_samples=4):
        self.num_samples = num_samples  # 要展示的重建样本数量
        self.save_path = save_path
        os.makedirs(save_path, exist_ok=True)  # 创建保存目录

    def on_validation_epoch_end(self, trainer, pl_module):
        """在验证 epoch 结束时执行一次重建可视化"""
        # 获取验证集的第一个 batch
        val_loader = trainer.val_dataloaders # 假设只有一个验证 DataLoader
        batch = next(iter(val_loader))
        x, _ = batch  # 假设 batch 是 (image, label) 的元组
        x = x[:self.num_samples].to(pl_module.device)  # 只取前几个样本

        # 使用 VAE 进行重建
        with torch.no_grad():
            pl_module.eval()
            x_reconstructed = pl_module([x])  # 假设你的 VAE 的 forward 返回重建结果
            pl_module.train()

        x_reconstructed = x_reconstructed[0].view(-1,28,28)

        # 归一化图像
        def normalize_image(img):
            return (img - img.min()) / (img.max() - img.min())

        x_norm = [normalize_image(img) for img in x]
        x_recon_norm = [normalize_image(img) for img in x_reconstructed]

        # 创建对比图：原始图像在上排，重建图像在下排
        fig, axes = plt.subplots(2, self.num_samples, figsize=(self.num_samples * 2, 4))
        
        # 绘制原始图像
        for i in range(self.num_samples):
            img = x_norm[i].cpu().numpy()[0]
            if len(img.shape) == 2:  # 单通道（如 MNIST）
                axes[0, i].imshow(img, cmap='gray')
            else:  # 多通道（如 CIFAR-10）
                axes[0, i].imshow(img.transpose(1, 2, 0))
            axes[0, i].set_title(f"Original {i+1}")
            axes[0, i].axis('off')

        # 绘制重建图像
        for i in range(self.num_samples):
            img = x_recon_norm[i].cpu().numpy()
            if len(img.shape) == 2:
                axes[1, i].imshow(img, cmap='gray')
            else:
                axes[1, i].imshow(img.transpose(1, 2, 0))
            axes[1, i].set_title(f"Recon {i+1}")
            axes[1, i].axis('off')

        plt.tight_layout()
        plt.savefig(f"{self.save_path}/val_reconstruction_epoch_{trainer.current_epoch:03d}.png")
        plt.close()

if __name__ == "__main__":

    epochs = 100
    batch_size = 64
    num_workers = 16
    expname = 'exp1'

    # 数据
    dm = MNISTDataModule( data_dir='/mnt/sda1/algorithom_code_summary/AIGC/01-VAE/data',  batch_size=batch_size, num_workers=16)
    
    # 模型
    model = MNISTVAE(32)

    # logger
    tb_logger = TensorBoardLogger(f"exp/{expname}/tblog", name="my_model")

    checkpoint_callback = ModelCheckpoint(
        dirpath=f"exp/{expname}/checkpoints/",
        filename="model-{epoch:02d}",  # 文件名包含 epoch 编号
        every_n_epochs=5,  # 每 5 个 epoch 保存一次
        save_top_k=-1,  # 保存所有检查点（-1 表示不限制数量）
    )

    vae_sample_callback = VAEReconstructionCallback(save_path = f'exp/{expname}/sampled_image', num_samples=3)

    trainer = pl.Trainer(
        max_epochs=epochs, # 训练的轮数
        accelerator='auto', # 加速
        devices='auto', # 训练的设备 优先使用所有的gpu
        logger=[tb_logger], #记录器，比较常用tbloger
        callbacks=[checkpoint_callback, vae_sample_callback], # 收集器，用于在batch/epochc层面收集想要的信息
        deterministic=True,
        enable_progress_bar=True,
        log_every_n_steps=10, # 记录频率
        fast_dev_run=False,  # 设为True可以快速检查代码是否能运行
        overfit_batches=0,  # 设为>0可以用于调试过拟合
    )

    trainer.fit(model, dm)