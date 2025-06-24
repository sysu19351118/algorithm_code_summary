import pytorch_lightning as pl
import torch.nn as nn
from torchmetrics import Accuracy
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau


class DualTowerRetrivelModel(pl.LightningModule):
    def __init__(self, learning_rate=1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.item_tower = None
        self.user_tower = None
        pass
    
    def forward(self, x):
        user_resp = self.user_tower(x)
        item_resp = self.item_tower(x)
        return user_resp, item_resp
    
    def training_step(self, batch, batch_idx):
        loss = None
        # point wise loss

        # pair wise loss

        # list wise loss

        return loss
    
    def validation_step(self, batch, batch_idx):
        pass
    
    def test_step(self, batch, batch_idx):
        pass
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        
        # 使用学习率调度器
        scheduler = {
            'scheduler': ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, verbose=True),
            'monitor': 'val_loss',
            'interval': 'epoch',
            'frequency': 1
        }
        return [optimizer], [scheduler]
