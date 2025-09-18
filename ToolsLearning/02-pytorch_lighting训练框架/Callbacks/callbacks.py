from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.callbacks import LearningRateMonitor

# 保存验证集准确率最高的前3个模型
checkpoint_callback = ModelCheckpoint(
    monitor="val_acc",
    dirpath="checkpoints/",
    filename="model-{epoch:02d}-{val_acc:.2f}",
    save_top_k=3,
    mode="max"
)

early_stop_callback = EarlyStopping(
    monitor="val_loss",
    patience=5,    # 容忍轮次
    mode="min"     # 监控指标需最小化
)

lr_monitor = LearningRateMonitor(logging_interval="epoch")
