## VAE 模型
基于pytorch lightning设计了一个VAE重建miniset的可调试流程

# VAE损失函数推导笔记

## 核心问题
目标是最大化边际似然：
$$ p_\theta(x) = \int p_\theta(x|z)p(z)dz $$

但直接计算困难（intractable），因此引入变分分布 $ q_\phi(z|x) $ 近似真实后验 $ p_\theta(z|x) $。

## 详细推导
[Notion Link](https://www.notion.so/VAE-2105e95ca88980c0a396ce855fc7635a?source=copy_link)

## 核心损失函数

```python
def loss_function(recon_x, x, mu, logvar):
    # 重构损失
    BCE = F.binary_cross_entropy_with_logits(recon_x, x.view(-1, 784), reduction='sum')
    
    # KL散度
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    return BCE + KLD