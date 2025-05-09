# 工作流程
<img src="images/image.png" alt="alt text" width="500">

# todolist
- [x] LightningModule
- [x] Trainer
- [x] LightningDataModule
- [x] Callbacks
- [ ] Loggers
- [ ] Advanced Features


```bash
# 运行方式
cd <your_path>/ToolsLearning/02-pytorch_lighting训练框架
python train.py
```


## 模块讲解

### LightningModule
见本文件夹下子目录

### Trainer
实例：
```python
trainer = pl.Trainer(
    max_epochs=20, # 训练的轮数
    accelerator='auto', # 加速
    devices='auto', # 训练的设备 优先使用所有的gpu
    logger=[tb_logger, csv_logger], #记录器，比较常用tbloger
    callbacks=[checkpoint_callback, early_stop_callback, lr_monitor], # 收集器，用于在batch/epochc层面收集想要的信息
    deterministic=True,
    enable_progress_bar=True,
    log_every_n_steps=10, # 记录频率
    fast_dev_run=False,  # 设为True可以快速检查代码是否能运行
    overfit_batches=0,  # 设为>0可以用于调试过拟合
)
```
pl.Trainer 是pl定义的训练、测试、验证一体化的框架，通过 
```python
trainer.fit(model, datamodule=dm)
trainer.test(model, datamodule=dm)
```
可以进入测试逻辑，使用 model 的 train_step 或者 test_step 完成模型的训练或者测试，并且会使用预先定义好的callbacks完成模型保存、中间结果的记录等等；相较于传统的训练框架，优点在于可以自动处理 optimizer.step()、loss.backward() 和 gradient clipping，无需手动编写训练循环。

#### 进阶：
trainer除了上述的基础功能外，还配置了很多训练优化的功能，极大的减轻了我们调参的负担，可以把更多的精力留给数据准备、模型开发阶段；
* 支持快速DDP分布式训练，不用编写复杂的ddp训练定义
```python
trainer = Trainer(accelerator="gpu", devices=4, strategy="ddp")
```
* 可以开启混合精度训练，提高显存利用率，加快训练速度
```python
trainer = Trainer(precision="16-mixed") 
```

*  梯度累积（Gradient Accumulation）模拟大 batch：在显存不足时，通过多步累积梯度再更新参数。
```python 
trainer = Trainer(accumulate_grad_batches=4) 
```


### LightningDataModule
见本文件夹下子目录


### Callbacks
见文件夹下的子目录

### Loggers

