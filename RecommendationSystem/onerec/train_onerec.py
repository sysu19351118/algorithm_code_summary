import json
import pandas as pd
import numpy as np
from collections import defaultdict
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import random
from transformers import BertTokenizer, BertModel
import pdb
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
import os
import pickle
from pytorch_lightning.strategies import DDPStrategy
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from typing import Dict, List, Tuple, Optional, Any
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import yaml

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from typing import Dict, List, Tuple, Optional, Any
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import yaml
import json
import pandas as pd
import numpy as np
from collections import defaultdict
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import random
from transformers import BertTokenizer, BertModel
import pdb
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from pytorch_lightning.callbacks import ModelCheckpoint
from model_dir.pl_onrec_model import OneRec
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from dataset.pl_dataset import OnerRecDataModule


def train(config):
    model = OneRec(config, )



    data_module = OnerRecDataModule(config, batch_size=config['train']['batch_size'], num_workers=config['train']['num_workers'], user_json_path = config['data']['user_data_path'], item_json_path = config['data']['item_data_path'])


    # 创建TensorBoard日志记录器
    logger = TensorBoardLogger(
        save_dir=config['train']['log_dir'],  # 日志保存目录，默认为'logs'
        name=config.get('experiment_name', 'dual_tower_model'),  # 实验名称
    )

    checkpoint_callback = ModelCheckpoint(
        every_n_epochs=5,           # 每1个epoch保存一次
        save_top_k=-1,              # 保存所有检查点
        filename='epoch-{epoch:02d}',
    )

    # 修改 Trainer 配置，添加 GPU 支持和TensorBoard日志记录器
    trainer = pl.Trainer(
        max_epochs=10,
        accelerator="gpu",      # 使用 GPU 加速器
        callbacks = [checkpoint_callback], 
        devices=8,              # 使用 1 个 GPU
        precision='bf16-mixed',           # 可选：使用混合精度训练以节省显存和加速
        logger=logger,          # 添加TensorBoard日志记录器
        log_every_n_steps=config['train']['log_freq'],   # 每10步记录一次日志
        strategy=DDPStrategy(find_unused_parameters=True)
    )

    trainer.fit(model, data_module)

    # trainer.test(model, data_module)
    return model, data_module

def load_config(config_path):
    """加载YAML配置文件"""
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

if __name__ == "__main__":
    config_file = '/home/tangzixuan.8/Generative_recall/generative_recall_rank/compare_exp/onerec/config/VG_onerec.yaml'
    config = load_config(config_file)
    model, data_module = train(config)