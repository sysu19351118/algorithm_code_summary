import numpy as np
from sklearn.datasets import make_blobs
from tqdm import tqdm
import cupy as cp

class KMeansGPU:
    def __init__(self, n_clusters=3, max_iter=300, random_state=42):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.random_state = random_state
        self.centroids = None
        self.labels = None
        self.cluster_centers_ = None  # 添加cluster_centers_属性
    
    def _initialize_centroids_kmeans_plusplus(self, X):
        """K-Means++ 初始化聚类中心"""
        cp.random.seed(self.random_state)
        centroids = cp.zeros((self.n_clusters, X.shape[1]))
        
        # 1. 随机选择第一个中心
        first_idx = cp.random.randint(X.shape[0])
        centroids[0] = X[first_idx]
        
        # 2. 逐个选择后续中心（概率与距离平方成正比）
        for i in range(1, self.n_clusters):
            # 计算每个点到最近中心的距离
            distances = cp.zeros(X.shape[0])
            for j in range(i):
                distances += cp.linalg.norm(X - centroids[j], axis=1) ** 2
            # 计算概率
            probs = distances / cp.sum(distances)
            
            # 手动实现带概率的随机选择
            cum_probs = cp.cumsum(probs)
            r = cp.random.rand()
            next_idx = cp.argmax(cum_probs >= r)
            
            centroids[i] = X[next_idx]
        
        return centroids
    
    def compute_distances(self, X, centroids):
        """GPU 加速的距离计算（欧氏距离）"""
        distances = cp.zeros((X.shape[0], self.n_clusters))
        for i in range(self.n_clusters):
            distances[:, i] = cp.linalg.norm(X - centroids[i], axis=1)
        return distances
    
    def find_closest_cluster(self, distances):
        """找到每个样本最近的聚类中心"""
        return cp.argmin(distances, axis=1)
    
    def _compute_centroids_with_empty_cluster_handling(self, X, labels):
        """计算新的聚类中心，并处理空簇"""
        centroids = cp.zeros((self.n_clusters, X.shape[1]))
        for i in range(self.n_clusters):
            mask = (labels == i)
            if cp.sum(mask) == 0:  # 空簇处理：随机选择一个点
                centroids[i] = X[cp.random.randint(X.shape[0])]
            else:
                centroids[i] = cp.mean(X[mask], axis=0)
        return centroids
    
    def fit(self, X):
        """训练 K-Means 模型"""
        # 将数据转移到 GPU
        X_gpu = cp.asarray(X, dtype=cp.float32)
        
        # 使用 K-Means++ 初始化聚类中心
        self.centroids = self._initialize_centroids_kmeans_plusplus(X_gpu)
        
        for _ in range(self.max_iter):
            # 计算距离并分配标签
            distances = self.compute_distances(X_gpu, self.centroids)
            self.labels = self.find_closest_cluster(distances)
            
            # 计算新中心（处理空簇）
            new_centroids = self._compute_centroids_with_empty_cluster_handling(X_gpu, self.labels)
            
            # 检查是否收敛
            if cp.allclose(self.centroids, new_centroids, rtol=1e-4):
                break
                
            self.centroids = new_centroids
        
        # 将结果转移回 CPU
        self.centroids = cp.asnumpy(self.centroids)
        self.labels = cp.asnumpy(self.labels)
        self.cluster_centers_ = self.centroids  # 设置cluster_centers_属性
    
    def predict(self, X):
        """预测新样本的聚类标签"""
        X_gpu = cp.asarray(X, dtype=cp.float32)
        centroids_gpu = cp.asarray(self.centroids, dtype=cp.float32)
        distances = self.compute_distances(X_gpu, centroids_gpu)
        return cp.asnumpy(self.find_closest_cluster(distances))
    
    def get_cluster_centers(self):
        """返回聚类中心"""
        return self.cluster_centers_
    
    def get_center_for_label(self, label):
        """返回指定标签对应的聚类中心"""
        if self.cluster_centers_ is None:
            raise ValueError("请先调用fit方法训练模型")
        if label < 0 or label >= self.n_clusters:
            raise ValueError(f"标签 {label} 超出范围 [0, {self.n_clusters-1}]")
        return self.cluster_centers_[label]

class MiniBatchKMeansGPU:
    def __init__(self, n_clusters=3, max_iter=100, batch_size=1_000_000, 
                 random_state=42, verbose=True):
        """
        Mini-Batch K-Means with GPU acceleration
        """
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.random_state = random_state
        self.verbose = verbose
        self.centroids = None
        self.counts = None
        self.cluster_centers_ = None  # 添加cluster_centers_属性
        
    def initialize_centroids(self, X):
        """使用随机样本初始化聚类中心"""
        cp.random.seed(self.random_state)
        random_idx = cp.random.permutation(X.shape[0])
        centroids = X[random_idx[:self.n_clusters]]
        return centroids
    
    def compute_distances(self, X, centroids):
        """GPU加速的距离计算"""
        distances = cp.zeros((X.shape[0], self.n_clusters))
        for i in range(self.n_clusters):
            distances[:, i] = cp.linalg.norm(X - centroids[i], axis=1)
        return distances
    
    def partial_fit(self, X_batch, centroids, counts):
        """
        处理一个mini-batch，更新聚类中心
        """
        # 计算距离并分配标签
        distances = self.compute_distances(X_batch, centroids)
        labels = cp.argmin(distances, axis=1)
        
        # 更新聚类中心
        new_centroids = cp.zeros_like(centroids)
        new_counts = cp.zeros(self.n_clusters, dtype=cp.int32)
        
        for i in range(self.n_clusters):
            mask = (labels == i)
            if cp.any(mask):
                new_centroids[i] = cp.mean(X_batch[mask], axis=0)
                new_counts[i] = cp.sum(mask)
        
        # 使用累积平均更新中心点
        for i in range(self.n_clusters):
            if new_counts[i] > 0:
                total = counts[i] + new_counts[i]
                centroids[i] = (centroids[i] * counts[i] + new_centroids[i] * new_counts[i]) / total
                counts[i] = total
                
        return centroids, counts
    
    def fit(self, X):
        """训练Mini-Batch K-Means模型"""
        # 将数据转移到GPU
        X_gpu = cp.asarray(X)
        n_samples = X_gpu.shape[0]
        
        # 初始化聚类中心
        self.centroids = self.initialize_centroids(X_gpu)
        self.counts = cp.zeros(self.n_clusters, dtype=cp.int32)
        
        # 进度条设置
        pbar = tqdm(total=self.max_iter, disable=not self.verbose)
        
        for iteration in range(self.max_iter):
            # 随机选择mini-batch
            indices = cp.random.randint(0, n_samples, self.batch_size)
            X_batch = X_gpu[indices]
            
            # 更新中心点
            self.centroids, self.counts = self.partial_fit(
                X_batch, self.centroids, self.counts)
            
            # 更新进度条
            pbar.update(1)
        
        pbar.close()
        
        # 将结果转移回CPU
        self.centroids = cp.asnumpy(self.centroids)
        self.cluster_centers_ = self.centroids  # 设置cluster_centers_属性
        return self
    
    def predict(self, X):
        """预测样本的聚类标签"""
        X_gpu = cp.asarray(X)
        centroids_gpu = cp.asarray(self.centroids)
        distances = self.compute_distances(X_gpu, centroids_gpu)
        return cp.asnumpy(cp.argmin(distances, axis=1))
    
    def get_cluster_centers(self):
        """返回聚类中心"""
        return self.cluster_centers_
    
    def get_center_for_label(self, label):
        """返回指定标签对应的聚类中心"""
        if self.cluster_centers_ is None:
            raise ValueError("请先调用fit方法训练模型")
        if label < 0 or label >= self.n_clusters:
            raise ValueError(f"标签 {label} 超出范围 [0, {self.n_clusters-1}]")
        return self.cluster_centers_[label]

class MyKmeans:
    def __init__(self, cluster, max_iter=100, random_state=42, batch_size=10000, mini_tresh=50000):
        self.kmeans = KMeansGPU(cluster, max_iter, random_state=random_state)
        self.minibatch_kmeans = MiniBatchKMeansGPU(cluster, max_iter, batch_size=batch_size)
        self.mini_tresh = mini_tresh
        self.current_model = None
        self.cluster_centers_ = None
    
    def fit(self, X):
        if X.shape[0] > self.mini_tresh:
            self.current_model = self.minibatch_kmeans
        else:
            self.current_model = self.kmeans

        self.current_model.fit(X)
        self.cluster_centers_ = self.current_model.cluster_centers_
    
    def predict(self, X):
        if self.current_model is None:
            raise ValueError("请先调用fit方法训练模型")
        return self.current_model.predict(X)
    
    @property
    def cluster_centers_(self):
        """返回聚类中心属性"""
        if self._cluster_centers_ is None:
            raise ValueError("请先调用fit方法训练模型")
        return self._cluster_centers_
    
    @cluster_centers_.setter
    def cluster_centers_(self, value):
        """设置聚类中心属性"""
        self._cluster_centers_ = value
    
    def get_cluster_centers(self):
        """返回所有聚类中心"""
        return self.cluster_centers_
    
    def get_center_for_label(self, label):
        """返回指定标签对应的聚类中心"""
        if self.cluster_centers_ is None:
            raise ValueError("请先调用fit方法训练模型")
        if label < 0 or label >= self.cluster_centers_.shape[0]:
            raise ValueError(f"标签 {label} 超出范围 [0, {self.cluster_centers_.shape[0]-1}]")
        return self.cluster_centers_[label]
    
    def get_centers_for_labels(self, labels):
        """返回多个标签对应的聚类中心"""
        if self.cluster_centers_ is None:
            raise ValueError("请先调用fit方法训练模型")
        labels = np.array(labels)
        if np.any((labels < 0) | (labels >= self.cluster_centers_.shape[0])):
            raise ValueError("存在超出范围的标签")
        return self.cluster_centers_[labels]


class MyKmeans:
    def __init__(self, cluster, max_iter=100, random_state=42, batch_size=10000, mini_tresh=50000):
        self.kmeans = KMeansGPU(cluster, max_iter, random_state=random_state)
        self.minibatch_kmeans = MiniBatchKMeansGPU(cluster, max_iter, batch_size=batch_size)
        self.mini_tresh = mini_tresh
        self.cluster_centers_ = None

    def fit(self, X):
        if  X.shape[0]>self.mini_tresh:
            kmeans = self.minibatch_kmeans
        else:
            kmeans = self.kmeans

        kmeans.fit(X)
        if  X.shape[0]>self.mini_tresh:
            self.cluster_centers_ = self.minibatch_kmeans.cluster_centers_
        else:
            self.cluster_centers_ = self.kmeans.cluster_centers_

    
    def predict(self, X):
        if  X.shape[0]>self.mini_tresh:
            kmeans = self.minibatch_kmeans
        else:
            kmeans = self.kmeans

        return kmeans.predict(X)



# 示例使用
if __name__ == "__main__":
    # 创建模拟数据
    # 生成3000万个64维随机向量（均匀分布在[0,1)区间）
    n_samples = 400_000
    n_features = 64
    X = np.random.rand(n_samples, n_features)
    print(X.shape)
    # 初始化并训练 K-Means
    kmeans = MyKmeans(cluster=100)
    kmeans.fit(X)
    res = kmeans.predict(X) # 计算聚类结果
    
 