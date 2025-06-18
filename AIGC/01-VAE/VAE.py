import pdb

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger
from torchmetrics import Accuracy
from torch.optim.lr_scheduler import ReduceLROnPlateau
# 设置随机种子以保证可重复性
torch.manual_seed(42)


    
class MNISTVAE(pl.LightningModule):
    def __init__(self, latent_dim, learning_rate=1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.latent_dim = latent_dim
        self.encoder = nn.Sequential(
            nn.Linear(784, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 2 * latent_dim)  # 输出均值和log方差
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 784),
            nn.Tanh()  # 输出在[-1,1]范围内，与归一化匹配
        )

        self.rec_loss = nn.MSELoss(reduction='sum')
        
    
    def forward(self, x):
        # 编码
        x = x[0].view(-1, 784)
        h = self.encoder(x)
        mu, logvar = h.chunk(2, dim=1)
        # 重参数化
        z = self.reparameterize(mu, logvar)
        # 解码
        x_recon = self.decoder(z)
        return x_recon, mu, logvar
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def sample(self, num_samples):
        # 从标准正态分布采样
        z = torch.randn(num_samples, self.latent_dim, device=self.device)
        # 通过解码器生成图像
        return self.decoder(z).view(-1, 1, 28, 28)  # 假设是MNIST 28x28图像

    

    def training_step(self, batch, batch_idx=None):
        x_recon, mu, logvar = self(batch)
        # 重建损失
        recon_loss = self.rec_loss(batch[0].reshape(batch[0].shape[0], -1), x_recon)
        # KL 散度损失 
        kl_loss = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=1).sum() 
        total_loss = recon_loss + kl_loss
        # 正确使用 self.log() 方法
        self.log("recon_loss", recon_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("kl_loss", kl_loss, on_step=True, on_epoch=True)
        self.log('train_loss', total_loss, on_step=True, on_epoch=True, prog_bar=True)
        
        return total_loss

    def validation_step(self, batch, batch_idx):
        pass

    def test_step(self, batch, batch_idx):
        pass
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        # 使用学习率调度器
        scheduler = {
            'scheduler': ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, verbose=True),
            'monitor': 'train_loss',
            'interval': 'epoch',
            'frequency': 1
        }
        return [optimizer], [scheduler]


if __name__ == "__main__":
    model = MNISTVAE(64)
    