import pdb
import pytorch_lightning as pl
from miniset_data_module import MNISTDataModule
from VAE import MNISTRQVAE
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import torch
import os

import os
import matplotlib.pyplot as plt
import torch
from pytorch_lightning.callbacks import Callback

class VAEVisualizationCallback(Callback):
    def __init__(self, save_path, num_samples=4):
        self.num_samples = num_samples
        self.save_path = save_path
        os.makedirs(save_path, exist_ok=True)
        self._has_validated = False  # Track if we've done validation this epoch

    def on_validation_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx=None):
        # Only run on first validation batch of each epoch
        if batch_idx == 0 and not self._has_validated:
            self._has_validated = True
            
            # Get batch data
            x, _ = batch
            x = x[:self.num_samples]  # Take first few samples
            
            # Generate reconstructions
            with torch.no_grad():
                pl_module.eval()
                x_recon, _, _, _, _ = pl_module(x)  # Adjust based on your RQ-VAE forward return
                pl_module.train()
            
            # Convert tensors to numpy
            x_np = x.cpu().numpy()
            x_recon_np = x_recon.view(-1, 1, 28, 28).cpu().numpy()
            
            # Create figure
            fig, axes = plt.subplots(2, self.num_samples, figsize=(self.num_samples*2, 4))
            
            # Plot original images (top row)
            for i in range(self.num_samples):
                axes[0, i].imshow(x_np[i].reshape(28, 28), cmap='gray')
                axes[0, i].set_title('Original')
                axes[0, i].axis('off')
            
            # Plot reconstructed images (bottom row)
            for i in range(self.num_samples):
                axes[1, i].imshow(x_recon_np[i].reshape(28, 28), cmap='gray')
                axes[1, i].set_title('Reconstructed')
                axes[1, i].axis('off')
            
            plt.suptitle(f'Epoch {trainer.current_epoch}')
            plt.tight_layout()
            
            # Save figure
            plt.savefig(f"{self.save_path}/epoch_{trainer.current_epoch:03d}.png")
            plt.close()
            
            # Optional: Log to TensorBoard
            if hasattr(trainer, 'logger') and trainer.logger is not None:
                trainer.logger.experiment.add_figure(
                    "Reconstructions",
                    fig,
                    global_step=trainer.global_step
                )

    def on_validation_epoch_end(self, trainer, pl_module):
        # Reset validation flag for next epoch
        self._has_validated = False


# 定义模型保存

if __name__ == "__main__":

    epochs = 100
    batch_size = 64
    num_workers = 16
    expname = 'exp1'

    # 数据
    dm = MNISTDataModule( data_dir='./data',  batch_size=batch_size, num_workers=16)
    
    # 模型
    model = MNISTRQVAE()

    # logger
    tb_logger = TensorBoardLogger(f"exp/{expname}/tblog", name="my_model")

    checkpoint_callback = ModelCheckpoint(
        dirpath=f"exp/{expname}/checkpoints/",
        filename="model-{epoch:02d}",  # 文件名包含 epoch 编号
        every_n_epochs=5,  # 每 5 个 epoch 保存一次
        save_top_k=-1,  # 保存所有检查点（-1 表示不限制数量）
    )

    vae_sample_callback = VAEVisualizationCallback(save_path = f'exp/{expname}/sampled_image', num_samples=16)

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