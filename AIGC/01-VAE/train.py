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
            
            # 将生成的样本保存为一张图片
            fig, axes = plt.subplots(1, self.num_samples, figsize=(self.num_samples*2, 2))
            for i, ax in enumerate(axes):
                ax.imshow(samples[i].cpu().numpy().reshape(28, 28), cmap='gray')
                ax.axis('off')
            
            plt.tight_layout()
            plt.savefig(f"{self.save_path}/epoch_{trainer.current_epoch:03d}.png")
            plt.close()
# 定义模型保存

if __name__ == "__main__":

    epochs = 100
    batch_size = 64
    num_workers = 16
    expname = 'exp1'

    # 数据
    dm = MNISTDataModule( data_dir='/mnt/sda1/algorithom_code_summary/AIGC/01-VAE/data',  batch_size=batch_size, num_workers=16)
    
    # 模型
    model = MNISTVAE(64)

    # logger
    tb_logger = TensorBoardLogger(f"exp/{expname}/tblog", name="my_model")

    checkpoint_callback = ModelCheckpoint(
        dirpath=f"exp/{expname}/checkpoints/",
        filename="model-{epoch:02d}",  # 文件名包含 epoch 编号
        every_n_epochs=5,  # 每 5 个 epoch 保存一次
        save_top_k=-1,  # 保存所有检查点（-1 表示不限制数量）
    )

    vae_sample_callback = VAESampleCallback(save_path = f'exp/{expname}/sampled_image', sample_interval=5, num_samples=4)

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