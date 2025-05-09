# Lightning Data Module
作用是为pl框架的训练、测试、验证环节准备数据

## 调用方式
通过trainer进行调用，例如
```python
dm = MNISTDataModule( ... ) # MNISTDataModule是我们自定义的ldm
trainer.fit(model, datamodule=dm)
```

调用之后，lightning data module会根据你的trainer的调用方式在 setup 函数中返回对应模式的dataset，在本案例中dataset是集称号的MINISET数据加载器，你也可以使用 torch.utils.data.Dataset 定义适合你的任务的数据加载模式

然后会根据你的trainer的调用方式，自动进入self.train_dataloader(), self.val_dataloader() 或者 self.test_dataloader(), 返回对应的dataloader

## 进阶
这里的dataloder是用的默认的data collation函数来把单个的数据聚合成batch数据，你可以自定义data collator函数来完成自定义的聚合，在NLP领域中用的会比较多。