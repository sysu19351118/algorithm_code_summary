import pdb
import numpy as np
import pandas as pd
from collections import defaultdict
import random
from typing import Dict, List, Tuple, Any
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


class UnifiedMovieLensDataset(Dataset):
    def __init__(self, data_path: str, min_rating: int = 4, 
                 num_pos: int = 1, num_easy_neg: int = 4, 
                 num_hard_neg: int = 2, max_seq_length: int = 50):
        """
        统一的双塔召回模型数据集
        
        参数:
            data_path: MovieLens 1M数据集路径
            min_rating: 视为正样本的最小评分
            num_pos: 每个用户采样的正样本数
            num_easy_neg: 每个正样本对应的简单负样本数
            num_hard_neg: 每个正样本对应的困难负样本数
            max_seq_length: 用户历史序列的最大长度
        """
        # 加载数据
        ratings = pd.read_csv(f"{data_path}/ratings.dat", sep="::", 
                             engine="python", 
                             names=["user_id", "movie_id", "rating", "timestamp"])
        
        movies = pd.read_csv(f"{data_path}/movies.dat", sep="::", 
                            engine="python", 
                            names=["movie_id", "title", "genres"],
                            encoding="latin1")
        
        # 构建数据结构
        self.user_pos_items = defaultdict(list)  # 用户的正样本
        self.user_all_items = defaultdict(list)  # 用户的所有交互(用于负采样)
        self.user_seq = defaultdict(list)         # 用户的历史行为序列(按时间排序)
        self.item_popularity = defaultdict(int)  # 物品流行度统计
        
        # 按时间排序的用户行为
        ratings = ratings.sort_values(by=["user_id", "timestamp"])
        
        for _, row in ratings.iterrows():
            user_id = row["user_id"]
            movie_id = row["movie_id"]
            rating = row["rating"]
            
            self.item_popularity[movie_id] += 1
            self.user_all_items[user_id].append(movie_id)
            
            if rating >= min_rating:
                self.user_pos_items[user_id].append(movie_id)
            
            # 构建用户序列(保留最后max_seq_length个)
            self.user_seq[user_id].append(movie_id)
            if len(self.user_seq[user_id]) > max_seq_length:
                self.user_seq[user_id] = self.user_seq[user_id][-max_seq_length:]
        
        # 所有物品和用户列表
        self.all_items = list(movies["movie_id"].unique())
        self.user_ids = list(self.user_pos_items.keys())
        
        # 参数设置
        self.num_pos = num_pos
        self.num_easy_neg = num_easy_neg
        self.num_hard_neg = num_hard_neg
        self.max_seq_length = max_seq_length
        
        # 准备热门物品列表(用于困难负采样)
        self.popular_items = sorted(self.item_popularity.items(), 
                                   key=lambda x: x[1], reverse=True)
        self.popular_items = [x[0] for x in self.popular_items]
        
        # 用户特征(可选: 可以添加更多用户特征)
        self.user_features = {
            uid: {
                "history_seq": seq,
                "history_len": len(seq)
            } 
            for uid, seq in self.user_seq.items()
        }
        
        # 物品特征(可选: 可以添加更多物品特征)
        self.item_features = {
            mid: {
                "popularity": self.item_popularity[mid],
                "genres": movies[movies["movie_id"] == mid]["genres"].values[0]
            }
            for mid in self.all_items
        }
    
    def __len__(self) -> int:
        return len(self.user_ids)
    
    def _sample_negative(self, user_id: int, num_samples: int, easy: bool = True) -> List[int]:
        """
        采样负样本
        
        参数:
            user_id: 用户ID
            num_samples: 需要采样的数量
            easy: 是否为简单负采样(True=简单负采样, False=困难负采样)
            
        返回:
            负样本列表
        """
        interacted = set(self.user_all_items[user_id])
        samples = []
        candidate_pool = self.all_items if easy else self.popular_items
        
        while len(samples) < num_samples:
            item = random.choice(candidate_pool)
            if item not in interacted and item not in samples:
                samples.append(item)
                if not easy and len(samples) >= num_samples:
                    break  # 困难负样本从热门物品中取
        
        return samples
    
    def __getitem__(self, index: int) -> Dict[str, Any]:
        """
        返回一个样本，包含:
        - 用户ID
        - 用户特征(历史序列等)
        - 正样本物品ID和特征
        - 简单负样本物品ID和特征
        - 困难负样本物品ID和特征
        """
        user_id = self.user_ids[index]
        
        # 采样正样本
        pos_items = random.sample(self.user_pos_items[user_id], 
                                min(self.num_pos, len(self.user_pos_items[user_id])))
        
        # 采样简单负样本
        easy_negs = self._sample_negative(user_id, self.num_easy_neg, easy=True)
        
        # 采样困难负样本
        hard_negs = self._sample_negative(user_id, self.num_hard_neg, easy=False)
        
        # 获取用户特征
        user_feat = {
            "user_id": user_id,
            "history_seq": self.user_seq[user_id],
            "history_len": len(self.user_seq[user_id])
        }
        
        # 获取物品特征
        def get_item_feats(item_ids):
            return {
                "item_ids": item_ids,
                "popularities": [self.item_features[i]["popularity"] for i in item_ids],
                "genres": [self.item_features[i]["genres"] for i in item_ids]
            }
        
        pos_feats = get_item_feats(pos_items)
        easy_neg_feats = get_item_feats(easy_negs)
        hard_neg_feats = get_item_feats(hard_negs)
        
        return {
            "user": user_feat,
            "pos_items": pos_feats,
            "easy_neg_items": easy_neg_feats,
            "hard_neg_items": hard_neg_feats
        }

def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    将多个样本合并成一个batch
    
    返回:
        包含以下字段的字典:
        - user_ids: 用户ID列表
        - history_seqs: 填充后的用户历史序列
        - history_lens: 用户历史序列实际长度
        - pos_items: 正样本物品ID列表
        - easy_neg_items: 简单负样本物品ID列表
        - hard_neg_items: 困难负样本物品ID列表
    """
    # 处理用户特征
    user_ids = [x["user"]["user_id"] for x in batch]
    history_seqs = [x["user"]["history_seq"] for x in batch]
    history_lens = [x["user"]["history_len"] for x in batch]
    
    # 填充历史序列
    max_len = max(history_lens)
    padded_seqs = []
    for seq in history_seqs:
        if len(seq) < max_len:
            padded_seq = seq + [0] * (max_len - len(seq))  # 用0填充
        else:
            padded_seq = seq
        padded_seqs.append(padded_seq)
    
    # 处理物品特征
    def process_items(key):
        item_ids = []
        for x in batch:
            item_ids.extend(x[key]["item_ids"])
        return item_ids
    
    pos_items = process_items("pos_items")
    easy_neg_items = process_items("easy_neg_items")
    hard_neg_items = process_items("hard_neg_items")
    
    return {
        "user_ids": torch.LongTensor(user_ids),
        "history_seqs": torch.LongTensor(padded_seqs),
        "history_lens": torch.LongTensor(history_lens),
        "pos_items": torch.LongTensor(pos_items),
        "easy_neg_items": torch.LongTensor(easy_neg_items),
        "hard_neg_items": torch.LongTensor(hard_neg_items)
    }

if __name__ == "__main__":

    dataset = UnifiedMovieLensDataset(
        data_path="/mnt/data2/zzixuantang/algorithm_code_summary/RecommendationSystem/movies_recommendation_demo/ml-1m",
        min_rating=4,        # 评分≥4视为正样本
        num_pos=2,           # 每个用户采样2个正样本
        num_easy_neg=4,      # 每个正样本对应4个简单负样本
        num_hard_neg=2,      # 每个正样本对应2个困难负样本
        max_seq_length=50    # 用户历史序列最大长度
    )
    # 创建DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=True,
        collate_fn=collate_fn
    )

    for batch in dataloader:
        pdb.set_trace()