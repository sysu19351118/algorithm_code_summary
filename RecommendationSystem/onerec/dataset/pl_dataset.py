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
import time
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


# 设置随机种子保证可重现
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


def split_train_test(df, user_hist):
    # 创建测试集：每个用户的最后一个交互
    test_records = []
    for user, items in tqdm(user_hist.items()):
        if len(items) > 0:
            last_item = items[-1]  # 最后一个交互的商品
            # 找到对应的行
            user_df = df[df['reviewerID'] == user]
            # 找到该用户最后一条关于last_item的记录（可能有多个相同商品的记录）
            last_record = user_df[user_df['asin'] == last_item].sort_values('unixReviewTime').iloc[-1]
            test_records.append(last_record)
    
    test_df = pd.DataFrame(test_records)
    
    # 创建训练集：排除测试集中的记录
    # 使用索引来排除测试集记录
    train_df = df.drop(test_df.index)
    return train_df, test_df



class AmazonDatasetLLada(Dataset):
    def __init__(self, config, df, user_hist, user2idx, item2idx, brand2idx, item_info_df, item_price, item_freq, semantic_id_dict, task_type):
        self.task_type = task_type
        self.df = df
        self.user_hist = user_hist
        self.user2idx = user2idx
        self.item2idx = item2idx
        self.config = config
        self.seq_max_len = config['model']['seq_max_len']
        self.tokenizer = BertTokenizer.from_pretrained('/home/tangzixuan.8/Generative_recall/bert-base-uncased')
        self.item_info_df = item_info_df
        self.brand2idx = brand2idx
        self.item_price = item_price
        self.semantic_id_dict = semantic_id_dict
        self.pad_sid = torch.zeros_like(list(self.semantic_id_dict.values())[0]).to(list(self.semantic_id_dict.values())[0].device)
        self.END_TOKEN = torch.tensor(config['model']['special_token']['END'])
        self.PAD_TOKEN = torch.tensor(config['model']['special_token']['PAD'])
        self.SEP_TOKEN = torch.tensor(config['model']['special_token']['SEP'])

        # 统计item id对不上的数量
        self.missing_item_count = 0
        self.wrong_price_count = 0
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        record = self.df.iloc[idx]
        
        # 用户侧信息
        user_info = {}
        user = record['reviewerID']
        
        item = record['asin'] # 用户当前交互的商品
        
        # 用户历史序列
        user_history = self.user_hist.get(user, [])

        
        try:
            # 查找项目最后一次出现的索引
            cutoff_index = len(user_history) - 1 - user_history[::-1].index(item)
            
            if cutoff_index == 0:
                cutoff_index += 1  # 这里暂时不考虑冷启动问题
            
            # 包含当前的token，+1是为了包含该项目本身
            cutoff_index = min(cutoff_index + 1, len(user_history))
            filtered_history = user_history[:cutoff_index]
            
        except ValueError:
            filtered_history = user_history.copy()

        # if self.task_type == "train":
        #     pdb.set_trace()

        user_history = filtered_history

        # 转换为索引
        user_idx = self.user2idx[user]

        # 处理用户历史行为
        item_idx = self.item2idx[item]
        hist_idx = []
        hist_idx_mask = []
        hist_sid = []

        for it in user_history:
            if it in self.item2idx:
                hist_idx.append(self.item2idx[it])
                if it in self.semantic_id_dict:
                    hist_sid.append(self.semantic_id_dict[it])
                else:
                    hist_sid.append(self.pad_sid)
                    # print("hist sid 缺失")
                hist_idx_mask.append(1)
            else:
                hist_idx.append(0)
                hist_idx_mask.append(0)
                hist_sid.append(self.pad_sid)
                
        hist_sid = torch.stack(hist_sid)
        # 将用户历史行为处理成sid
        
        user_info["user_idx"] = user_idx
        user_info["hist_idx"] = hist_idx
        user_info["hist_idx_mask"] = hist_idx_mask
        user_info['hist_sid'] = hist_sid


        # 搞成token形式
        user_history_tokens = []
        for valid_index in range(sum(hist_idx_mask)):
            user_history_tokens.append(self.SEP_TOKEN)
            user_history_tokens.append(hist_sid[valid_index])
        # if len(user_history_tokens) > 0:
        #     user_history_tokens[-1] = self.END_TOKEN
        # else:
        #     user_history_tokens.append(self.END_TOKEN)
        user_info['user_hist_token'] =  torch.cat([t.view(-1) for t in user_history_tokens])
        return user_info

    def collate_fn(self, batch):
        batch_user_idx = []
        batch_hist_idx = []
        batch_hist_idx_mask = []
        batch_hist_sid = []
        batch_hist_sid_tokens = []
        max_token_lenth = 0
        for usr_info in batch:
            # 用户信息
            batch_user_idx.append(usr_info['user_idx'])
            batch_hist_idx.append(usr_info['hist_idx'])
            batch_hist_idx_mask.append( sum( usr_info['hist_idx_mask'] ) * ( self.config['model']['sid_lenth'] + 1 ) )
            batch_hist_sid.append(usr_info['hist_sid'])
            max_token_lenth = max(max_token_lenth, usr_info['user_hist_token'].shape[0])

        # 对max token lenth 进行填充
        for usr_info in batch:
            term_user_hist_token = usr_info['user_hist_token']
            add_token  = [self.PAD_TOKEN]*(max_token_lenth - term_user_hist_token.shape[0])
            term_user_hist_token = add_token + [term_user_hist_token]
            term_user_hist_token = torch.cat([t.view(-1) for t in term_user_hist_token])
            batch_hist_sid_tokens.append(term_user_hist_token)

        # 对齐user history长度
        max_hist_len = max([len(h) for h in batch_hist_idx])
        batch_hist_idx = [([0] * (max_hist_len - len(h)) + h) if len(h) < max_hist_len else h for h in batch_hist_idx]

        batch_user_idx = torch.tensor(batch_user_idx, dtype=torch.long)
        batch_hist_idx = torch.tensor(batch_hist_idx, dtype=torch.long)
        # batch_hist_sid = torch.stack(batch_hist_sid, dim=0) 
        batch_hist_sid_tokens = torch.stack(batch_hist_sid_tokens, dim=0)

        batch_size = batch_hist_sid_tokens.shape[0]
        sqlen = batch_hist_sid_tokens.shape[1]
        batch_hist_idx_arange = torch.arange( batch_hist_sid_tokens.shape[1] ).unsqueeze(dim=0).repeat(batch_size, 1)
        batch_hist_idx_mask = batch_hist_idx_arange >= (sqlen - torch.tensor(batch_hist_idx_mask)).unsqueeze(dim=1)

        

        user_feature = {
            'user_idx': batch_user_idx,
            'hist_idx': batch_hist_idx,
            'hist_idx_mask': batch_hist_idx_mask,
            'hist_token': batch_hist_sid_tokens
        }
        return user_feature


# 示例数据模块
class OnerRecDataModule(pl.LightningDataModule):
    def __init__(
        self,
        config: Dict[str, Any] = None,
        batch_size: int = 256,
        num_workers: int = 0,
        user_json_path: str = '/mnt/data2/zzixuantang/ICLR/data/amazon-toy/Toys_and_Games.json',
        item_json_path: str = '/mnt/data2/zzixuantang/ICLR/data/amazon-toy/meta_Toys_and_Games.json',
        optimizer: str = "adam",
        scheduler: str = "step",
        scheduler_params: Optional[Dict[str, Any]] = 'cosine'
    ):
        super().__init__()
        self.config = config
        self.batch_size = batch_size
        self.num_workers = num_workers
        df, item_df, user_hist, user2idx, item2idx, brand2idx, item_price, item_freq, self.train_df, self.test_df, semantic_id_dict  = self.load_feature(config, user_json_path, item_json_path)
        self.df = df
        self.item_df = item_df
        self.user_hist = user_hist  
        self.user2idx = user2idx
        self.item2idx = item2idx
        self.brand2idx = brand2idx
        self.item_price = item_price
        self.item_freq = item_freq
        self.semantic_id_dict = semantic_id_dict

    # 读取数据
    def load_data(self, file_path, sample_frac=1.0):
        reviews = []
        with open(file_path, 'r') as f:
            for line in f:
                reviews.append(json.loads(line))
        print('用户数据load完成')
        df = pd.DataFrame(reviews)
        if sample_frac < 1.0:
            df = df.sample(frac=sample_frac, random_state=SEED)
        return df

    def load_item_data(self, file_path):
        items = []
        with open(file_path, 'r') as f:
            for line in f:
                items.append(json.loads(line))
        item_df = pd.DataFrame(items)
        print('物品数据load完成')
        return item_df




    # 在你的load_feature函数中使用：
    def load_feature(self, config, user_json_path, item_json_path):

        # semantic id
        semantic_dict = {}

        sid_data = torch.load(config['data']['semantic_id_paths'])
        asin_list = sid_data['original_asins']
        sid_tensor = sid_data['semantic_ids']

        for asin, sid in zip(asin_list, sid_tensor):
            semantic_dict[asin] = torch.tensor(sid)




        df = self.load_data(user_json_path, sample_frac=1)  
        item_df = self.load_item_data(item_json_path)
        # 转换时间戳
        df['unixReviewTime'] = pd.to_numeric(df['unixReviewTime'])
        df['reviewTime'] = pd.to_datetime(df['unixReviewTime'], unit='s')

        # 计算每个用户的交互次数
        user_interaction_count = df.groupby('reviewerID').size()

        # 筛选出交互次数 >= 3 的用户
        valid_users = user_interaction_count[user_interaction_count >= 5].index

        # 过滤df，只保留有效用户的记录
        df = df[df['reviewerID'].isin(valid_users)].copy()
        item_df = item_df[item_df['asin'].isin(df['asin'].unique())]

        # 重新构建用户历史序列（基于过滤后的df）
        user_hist = df.groupby('reviewerID').apply(
            lambda x: x.sort_values('unixReviewTime')['asin'].tolist()
        ).to_dict()

        # 划分训练集和测试集
        

        # 缓存路径
        cache_dir = config['data']['cache_path']
        train_cache = os.path.join(cache_dir, "train_df.pkl")
        test_cache = os.path.join(cache_dir, "test_df.pkl")

        # 检查缓存是否存在
        if os.path.exists(train_cache) and os.path.exists(test_cache):
            print("从缓存加载数据...")
            with open(train_cache, 'rb') as f:
                train_df = pickle.load(f)
            with open(test_cache, 'rb') as f:
                test_df = pickle.load(f)
        else:
            print("缓存不存在，创建数据并保存...")
            train_df, test_df = split_train_test(df, user_hist)
            
            # 确保目录存在
            os.makedirs(cache_dir, exist_ok=True)
            
            # 保存到缓存
            with open(train_cache, 'wb') as f:
                pickle.dump(train_df, f)
            with open(test_cache, 'wb') as f:
                pickle.dump(test_df, f)
            print("数据已保存到缓存")

        # 创建商品ID到索引的映射（使用完整df的商品集合）
        item_ids = np.unique(np.concatenate([df['asin'].unique(), item_df['asin'].unique()]))
        item2idx = {item: idx for idx, item in enumerate(item_ids)}
        num_items = len(item2idx)

        # 统计item的频次（基于训练集）
        freq = [0] * len(item_ids)
        for item_id in train_df['asin']:
            if item_id in item2idx:
                idx = item2idx[item_id]
                freq[idx] += 1

        # 创建商品品牌到索引的映射
        # brand_ids = item_df['brand'].unique()
        # brand2idx = {brand: idx for idx, brand in enumerate(brand_ids)}
        # num_brands = len(brand2idx)

        # 创建用户ID到索引的映射
        user_ids = df['reviewerID'].unique()
        user2idx = {user: idx for idx, user in enumerate(user_ids)}
        num_users = len(user2idx)

        item_price = None
        brand2idx = None

        return df, item_df, user_hist, user2idx, item2idx, brand2idx, item_price, freq, train_df, test_df, semantic_dict

    def prepare_data(self):
        """下载或准备数据"""
        pass
    
    def setup(self, stage: Optional[str] = None):
        """设置数据集"""
        pass
    
    def train_dataloader(self):
        """训练数据加载器"""
        train_dataset = AmazonDatasetLLada(self.config, self.train_df, self.user_hist, self.user2idx, self.item2idx, self.brand2idx, self.item_df, self.item_price, self.item_freq, self.semantic_id_dict, task_type = "train")
        return DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, collate_fn=train_dataset.collate_fn)
    
    def val_dataloader(self):
        """验证数据加载器"""
        test_dataset = AmazonDatasetLLada(self.config, self.test_df, self.user_hist, self.user2idx, self.item2idx, self.brand2idx, self.item_df, self.item_price, self.item_freq, self.semantic_id_dict, task_type = "test")
        return DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, collate_fn=test_dataset.collate_fn)
    
    def test_dataloader(self):
        """测试数据加载器"""
        test_dataset = AmazonDatasetLLada(self.config, self.test_df, self.user_hist, self.user2idx, self.item2idx, self.brand2idx, self.item_df, self.item_price, self.item_freq, self.semantic_id_dict, task_type = "test")
        return DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, collate_fn=test_dataset.collate_fn)


def load_config(config_path):
    """加载YAML配置文件"""
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


if __name__ == "__main__":
    config = load_config('/home/tangzixuan.8/Generative_recall/generative_recall_rank/compare_exp/onerec/config/toys_onerec.yaml')
    data_module = OnerRecDataModule(config, batch_size=config['train']['batch_size'], num_workers=config['train']['num_workers'], user_json_path = config['data']['user_data_path'], item_json_path = config['data']['item_data_path'])
    train_dataloader = data_module.train_dataloader()

    for batch in train_dataloader:
        pdb.set_trace()