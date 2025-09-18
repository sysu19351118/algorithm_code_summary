# Lightning Module
实现了一个例子用于手写数字分类
```python
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

```

## 讲解：
* __init__ 定义逻辑基本和nn.Module一致， 在 中定义class的全局变量，完成模块定义；
* forward函数中定义模型的计算逻辑
* training_step 实现前向传播和loss计算，trainer.fit( )启动训练时， 所有的batch自动进入此逻辑
* test_step 完成模型的测试
* configure_optimizers 定义优化器和衰减器

## 与nn.Module的不同点：
由于传统torch框架需要自定义训练逻辑，所以需要在自行实现的train函数中使用实例化好的model完成前向传播和loss计算，训练过程和模型是解耦的，而Lightning Module将训练过程和模型定义耦合了起来，并且Lightning Module不需要我们去维护梯度，只需要在train_step中定义好训练逻辑并且返回loss即可