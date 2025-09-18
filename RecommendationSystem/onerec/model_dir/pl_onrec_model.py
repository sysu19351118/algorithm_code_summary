
import sys

# sys.path.append('/home/tangzixuan.8/Generative_recall/LLaDA_weight')
import torch
import numpy as np
import torch.nn.functional as F
from transformers import AutoConfig
import pdb
from transformers import AutoTokenizer, AutoModel
import copy
import os
import pickle
import yaml
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from typing import Optional, Dict, Any, List
import torchmetrics
from model_dir.arm_transformer import OneRecModel
import math

class OneRec(pl.LightningModule):
    def __init__(
        self,
        config,
    ):
        """
        初始化Lightning模型
        
        Args:
            input_dim: 输入维度
            hidden_dim: 隐藏层维度
            output_dim: 输出维度
            learning_rate: 学习率
            dropout_rate: dropout比率
        """
        super().__init__()
        device = 'cuda'
        self.model = OneRecModel(
            config
        )
        self.config = config
        self.loss_fn = torch.nn.CrossEntropyLoss(reduction='none')
        sid_count_path = config['data']['sid_count_path']
        with open(sid_count_path, 'r') as f:
            import json
            self.sid_count = json.load(f)
        # pdb.set_trace()
        self.mean_lenth = int(sum(self.sid_count.values())/ len(self.sid_count.keys()))+1
        print(self.mean_lenth)

        self.hit_rate_at_k = [0, 0, 0, 0, 0, 0] # 1, 5, 10, 15, 20, sum
        self.ndcg_at_k = [0, 0, 0, 0, 0, 0] # 1, 5, 10, 15, 20, sum

    def forward(self, user_repr, item_token):
        pass

    def change_word(self, input_ids, mask_or_change_indices):
        
        return input_ids, masked_indices
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.config['train']['learning_rate'],
            weight_decay=self.config['train']['weight_decay'],
            eps=1e-8
        )
        
        # 带重启的余弦退火调度器
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=self.config['train'].get('restart_epochs', 10),  # 第一次重启的周期
            T_mult=self.config['train'].get('restart_multiplier', 2),  # 重启周期倍增因子
            eta_min=self.config['train'].get('min_learning_rate', 1e-6)
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step',   # 每个训练step后更新（更适合WarmRestarts）
                'frequency': 1,
            }
        }


    def training_step(self, batch, batch_idx, eps = 1e-6):
        logits, moe_loss = self.model(batch['hist_token'][:, :-4], batch['hist_token'][:, -4:])
        logits = logits[:, :3, :]
        logits = logits.reshape(-1, logits.shape[-1])
        gt = batch['hist_token'][:, -3:].reshape(-1)
        loss = self.loss_fn(
            logits,
            gt
        ).sum(dim=-1).mean()

        total_loss = loss + moe_loss
        self.log('train_loss', loss, prog_bar=True)
        self.log('moe_loss', moe_loss, prog_bar=True)
        return total_loss
    
    def validation_step(self, batch, batch_idx):
        self.model
        pass

    import math

    def ndcg_at_k_single_positive(self, rank, k):
        """
        计算只有一个正样本时的NDCG@K
        
        参数:
        rank (int): 正样本在推荐列表中的位置（从1开始计数）
        k (int): 要考虑的top K个项目
        
        返回:
        float: NDCG@K 值，范围 [0, 1]
        """
        # 如果正样本不在前K个中，NDCG@K为0
        if rank > k or rank < 1:
            return 0.0
        
        # 计算DCG@K：只有一个相关项，位置在rank处
        # 相关项的相关度得分为1（因为是正样本），其他项为0
        dcg = 1.0 / math.log2(rank + 1)  # 通常使用log2(rank+1)避免除0
        
        # 计算IDCG@K：理想情况是正样本排在第一位
        idcg = 1.0 / math.log2(1 + 1)  # 即 1.0
        
        return dcg / idcg

    
    def test_step(self, batch, batch_idx):
        
        batch_size = batch['hist_token'].shape[0]
        for b in range(batch_size):
            beam_search_result = self.model.generate(batch['hist_token'][b:b+1,:-4], 256, 4, 10)
            gt = batch['hist_token'][b:b+1, -4:]
            key = "-".join(str(i) for i in gt.tolist()[0][1:])
            valid_candidate_sum = [0]
            hit_index = -1
            for index, gen_result in enumerate(beam_search_result):
                gen_key = "-".join(str(i) for i in gen_result[1:])
                try:
                    valid_candidate_sum.append(self.sid_count[gen_key])
                except:
                    valid_candidate_sum.append(2)
                if gen_key == key:
                    hit_index = index
                    break
            
            if hit_index != -1:
                first_index = sum(valid_candidate_sum[:hit_index+1])
                last_index = sum(valid_candidate_sum[:hit_index+2])

                for index, mount in enumerate([1,5,10,15,20]):
                    
                    if mount >= last_index:
                        score = 1
                    elif mount < last_index:
                        score = 0
                    else:
                        score = (count - first_index) / (last_index - first_index)
                    self.hit_rate_at_k[index] += score
                    self.ndcg_at_k[index] += self.ndcg_at_k_single_positive((first_index+ last_index)/2, mount)
            self.hit_rate_at_k[-1]+=1
            self.ndcg_at_k[-1]+=1

            term_sum = self.hit_rate_at_k[-1]
            
            print(f"hit@1:{self.hit_rate_at_k[0]/term_sum}; hit@5:{self.hit_rate_at_k[1]/term_sum}; hit@10:{self.hit_rate_at_k[2]/term_sum}; hit@15:{self.hit_rate_at_k[3]/term_sum}; hit@20:{self.hit_rate_at_k[4]/term_sum}")
            print(f"ndcg@1:{self.ndcg_at_k[0]/term_sum}; ndcg@5:{self.ndcg_at_k[1]/term_sum}; ndcg@10:{self.ndcg_at_k[2]/term_sum}; ndcg@15:{self.ndcg_at_k[3]/term_sum}; ndcg@20:{self.ndcg_at_k[4]/term_sum}")
            print("\n\n")
    
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        pass
    
    def on_train_epoch_end(self):
        pass
    
    def on_validation_epoch_end(self):
        pass




def main(config):
    model = OneRecModel(config).cuda()
    


def load_config(config_path):
    """加载YAML配置文件"""
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config
    # amp_bf16

if __name__ == "__main__":
    config_file = '/home/tangzixuan.8/Generative_recall/generative_recall_rank/compare_exp/onerec/config/toys_onerec.yaml'
    config = load_config(config_file)
    main(config)
