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


class MNISTClassifier(pl.LightningModule):
    def __init__(self, learning_rate=1e-3):
        super().__init__()
        self.save_hyperparameters()
        
        # 网络结构
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)
        
        # 指标
        self.train_accuracy = Accuracy(task='multiclass', num_classes=10)
        self.val_accuracy = Accuracy(task='multiclass', num_classes=10)
        self.test_accuracy = Accuracy(task='multiclass', num_classes=10)
    
    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        
        # 记录训练指标
        preds = torch.argmax(logits, dim=1)
        self.train_accuracy(preds, y)
        
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_acc', self.train_accuracy, on_step=False, on_epoch=True, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        
        preds = torch.argmax(logits, dim=1)
        self.val_accuracy(preds, y)
        
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', self.val_accuracy, prog_bar=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        
        preds = torch.argmax(logits, dim=1)
        self.test_accuracy(preds, y)
        
        self.log('test_loss', loss, prog_bar=True)
        self.log('test_acc', self.test_accuracy, prog_bar=True)
        return loss
    
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
