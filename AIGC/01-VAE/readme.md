## VAE 模型
基于pytorch lightning设计了一个VAE重建miniset的可调试流程
# VAE损失函数推导笔记

## 核心问题
目标是最大化边际似然：
$$ p_\theta(x) = \int p_\theta(x|z)p(z)dz $$

但直接计算困难（intractable），因此引入变分分布 $ q_\phi(z|x) $ 近似真实后验 $ p_\theta(z|x) $。

## 详细推导

### 1. 从KL散度出发
$$ D_{KL}(q_\phi(z|x) \| p_\theta(z|x)) = \mathbb{E}_{q}[\log q_\phi(z|x) - \log p_\theta(z|x)] $$

### 2. 展开真实后验
根据贝叶斯定理：
$$ p_\theta(z|x) = \frac{p_\theta(x|z)p(z)}{p_\theta(x)} $$

代入KL散度：
$$
\begin{aligned}
D_{KL} &= \mathbb{E}_{q}[\log q_\phi(z|x) - \log \frac{p_\theta(x|z)p(z)}{p_\theta(x)}] \\
&= \mathbb{E}_{q}[\log q_\phi(z|x) - \log p_\theta(x|z) - \log p(z) + \log p_\theta(x)]
\end{aligned}
$$

### 3. 整理得到关键等式
$$ \log p_\theta(x) = \underbrace{\mathbb{E}_{q}[\log p_\theta(x|z)] - D_{KL}(q_\phi(z|x) \| p(z))}_{\text{ELBO}} + D_{KL}(q_\phi(z|x) \| p_\theta(z|x)) $$

### 4. 证据下界(ELBO)
因为KL散度非负：
$$ \log p_\theta(x) \geq \text{ELBO} = \mathbb{E}_{q}[\log p_\theta(x|z)] - D_{KL}(q_\phi(z|x) \| p(z)) $$

## 损失函数组成

### 1. 重构损失（负对数似然）
$$ \mathcal{L}_{\text{recon}} = -\mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x|z)] $$

实际计算：
- 二值数据（如MNIST）：二元交叉熵
- 连续数据：均方误差(MSE)

### 2. KL散度项
$$ \mathcal{L}_{\text{KL}} = D_{KL}(q_\phi(z|x) \| p(z)) $$

当 $ p(z) = \mathcal{N}(0,I) $ 且 $ q_\phi(z|x) = \mathcal{N}(\mu,\sigma^2I) $ 时：
$$ D_{KL} = \frac{1}{2}\sum_{j=1}^J (\mu_j^2 + \sigma_j^2 - \log\sigma_j^2 - 1) $$

### 3. 最终损失函数
$$ \mathcal{L}(\theta,\phi;x) = \mathcal{L}_{\text{recon}} + \mathcal{L}_{\text{KL}} $$

## 为什么需要变分分布？

1. **计算可行性**：
   - 真实后验 $ p_\theta(z|x) $ 难以计算
   - 通过 $ q_\phi(z|x) $ 近似，使ELBO可优化

2. **理论保证**：
   - ELBO是 $ \log p_\theta(x) $ 的下界
   - 最大化ELBO → 间接最大化似然

3. **实现方式**：
   - 编码器输出 $ q_\phi(z|x) $ 的参数（均值μ和方差σ²）
   - 使用重参数化技巧实现可微采样：
     $$ z = \mu + \sigma \odot \epsilon, \quad \epsilon \sim \mathcal{N}(0,I) $$

## 代码实现（PyTorch示例）

```python
def loss_function(recon_x, x, mu, logvar):
    # 重构损失
    BCE = F.binary_cross_entropy_with_logits(recon_x, x.view(-1, 784), reduction='sum')
    
    # KL散度
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    return BCE + KLD