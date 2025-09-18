import torch
import numpy as np
from sklearn.cluster import KMeans
from tqdm import tqdm
import pickle
import json
from my_kmeans import MyKmeans
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

class RQKMeans:
    """Residual Quantized K-Means 实现"""
    
    def __init__(self, n_levels=3, n_clusters=256, random_state=42):
        self.n_levels = n_levels
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.kmeans_models = []  # 每层的KMeans模型
        self.cluster_centers = []  # 每层的聚类中心
        
    def fit(self, embeddings):
        """训练RQKMeans模型"""
        print(f"训练RQKMeans模型，层级数: {self.n_levels}, 每层聚类数: {self.n_clusters}")
        
        residuals = embeddings.copy()
        
        for level in range(self.n_levels):
            print(f"训练第 {level + 1} 层...")
            
            # 训练KMeans
            kmeans = MyKmeans(
                cluster=self.n_clusters,
                random_state=self.random_state,
            )
            kmeans.fit(residuals)
            
            # 保存模型和聚类中心
            self.kmeans_models.append(kmeans)
            self.cluster_centers.append(kmeans.cluster_centers_)
            
            # 计算残差（如果是最后一层就不需要了）
            if level < self.n_levels - 1:
                # 获取每个样本的聚类分配
                labels = kmeans.predict(residuals)
                # 计算残差：原始向量 - 聚类中心
                residuals = residuals - kmeans.cluster_centers_[labels]
                print(f"第 {level + 1} 层残差范数均值: {np.mean(np.linalg.norm(residuals, axis=1)):.4f}")
        
        return self
    
    def predict(self, embeddings):
        """预测semantic IDs"""
        print("生成semantic IDs...")
        
        n_samples = embeddings.shape[0]
        semantic_ids = np.zeros((n_samples, self.n_levels), dtype=int)
        residuals = embeddings.copy()
        
        for level in range(self.n_levels):
            # 预测当前层的聚类
            labels = self.kmeans_models[level].predict(residuals)
            semantic_ids[:, level] = labels
            
            # 计算残差（最后一层不需要）
            if level < self.n_levels - 1:
                residuals = residuals - self.kmeans_models[level].cluster_centers_[labels]
        
        return semantic_ids
    
    def save(self, filepath):
        """保存模型"""
        with open(filepath, 'wb') as f:
            pickle.dump({
                'n_levels': self.n_levels,
                'n_clusters': self.n_clusters,
                'random_state': self.random_state,
                'kmeans_models': self.kmeans_models,
                'cluster_centers': self.cluster_centers
            }, f)
        print(f"模型已保存至: {filepath}")
    
    @classmethod
    def load(cls, filepath):
        """加载模型"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        model = cls(data['n_levels'], data['n_clusters'], data['random_state'])
        model.kmeans_models = data['kmeans_models']
        model.cluster_centers = data['cluster_centers']
        return model

def compute_semantic_ids(embeddings_path, output_path, n_levels=3, n_clusters=256):
    """
    计算semantic IDs的主函数
    
    Args:
        embeddings_path: BERT嵌入文件路径
        output_path: 输出文件路径
        n_levels: RQ层级数
        n_clusters: 每层聚类数
    """
    # 1. 加载BERT嵌入
    print("加载BERT嵌入...")
    data = torch.load(embeddings_path)
    embeddings = data['embeddings'].numpy()
    asins = data['asins'].numpy()
    original_asins = data.get('original_asins', [str(asin) for asin in asins])
    
    print(f"嵌入形状: {embeddings.shape}")
    print(f"商品数量: {len(asins)}")
    
    # 2. 数据预处理
    print("数据预处理...")
    # 归一化嵌入向量
    embeddings_norm = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    
    # 3. 训练RQKMeans模型
    rqkmeans = RQKMeans(n_levels=n_levels, n_clusters=n_clusters)
    rqkmeans.fit(embeddings_norm)
    
    # 4. 生成semantic IDs
    semantic_ids = rqkmeans.predict(embeddings_norm)
    
    # 5. 保存结果
    print("保存结果...")
    result = {
        'asins': asins,
        'original_asins': original_asins,
        'semantic_ids': semantic_ids,
        'embeddings': embeddings,  # 可选：保存原始嵌入
        'rqkmeans_config': {
            'n_levels': n_levels,
            'n_clusters': n_clusters,
            'random_state': 42
        },
        'bert_model_info': {
            'model_name': data.get('model_name', 'bert-base-uncased'),
            'pooling_strategy': data.get('pooling_strategy', 'cls')
        }
    }
    
    # 保存为pt文件
    torch.save(result, output_path)
    
    # 同时保存为JSON格式便于查看
    json_output_path = output_path.replace('.pt', '_metadata.json')
    with open(json_output_path, 'w', encoding='utf-8') as f:
        json.dump({
            'num_items': len(asins),
            'embedding_dim': embeddings.shape[1],
            'n_levels': n_levels,
            'n_clusters': n_clusters,
            'semantic_id_shape': semantic_ids.shape,
            'first_10_examples': [
                {
                    'asin': str(original_asins[i]),
                    'semantic_id': semantic_ids[i].tolist()
                }
                for i in range(min(10, len(asins)))
            ]
        }, f, indent=2, ensure_ascii=False)
    
    print(f"Semantic IDs计算完成！")
    print(f"结果已保存至: {output_path}")
    print(f"元数据已保存至: {json_output_path}")
    print(f"Semantic IDs形状: {semantic_ids.shape}")
    print("\n前5个商品的semantic IDs:")
    for i in range(min(5, len(asins))):
        print(f"ASIN: {original_asins[i]}, Semantic ID: {semantic_ids[i]}")
    
    return result

def analyze_semantic_ids(semantic_ids_path):
    """分析semantic IDs的分布"""
    print("分析semantic IDs分布...")
    data = torch.load(semantic_ids_path)
    semantic_ids = data['semantic_ids']
    
    print(f"总商品数: {semantic_ids.shape[0]}")
    print(f"层级数: {semantic_ids.shape[1]}")
    
    for level in range(semantic_ids.shape[1]):
        unique_ids = np.unique(semantic_ids[:, level])
        print(f"第 {level + 1} 层 - 唯一ID数: {len(unique_ids)}, 范围: [{unique_ids.min()}, {unique_ids.max()}]")
    
    return data

# 使用示例
if __name__ == "__main__":
    # 配置参数
    EMBEDDINGS_PATH = "/home/tangzixuan.8/Generative_recall/generative_recall_rank/data/5-core/Arts_Crafts_and_Sewing/OneRecCache/sid/ACS_item_embeddings.pt"  # 你的BERT嵌入文件
    OUTPUT_PATH = "/home/tangzixuan.8/Generative_recall/generative_recall_rank/data/5-core/Arts_Crafts_and_Sewing/OneRecCache/sid/rqkmeans_semantic_ids.pt"
    N_LEVELS = 3      # 三层semantic ID
    N_CLUSTERS = 256  # 每层256个聚类（可以调整）
    
    # 计算semantic IDs
    result = compute_semantic_ids(
        embeddings_path=EMBEDDINGS_PATH,
        output_path=OUTPUT_PATH,
        n_levels=N_LEVELS,
        n_clusters=N_CLUSTERS
    )
    
