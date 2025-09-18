# DeepSeek

# MMoE
MMoE 是 MoE (Mixture-of-Experts) 的一个经典变体，旨在为多任务学习量身定制。其核心思想是：为每个任务配备一个独立的“门控网络”（Gate），让每个任务都能自主地选择组合最适合自己的“专家”（Expert），从而更灵活地在任务间共享信息或保留独特性。
核心组成：
* 专家网络（Experts）：n 个共享的子网络（通常是全连接层+激活函数）。每个专家对输入进行处理，产生自己的输出。
* 门控网络（Gates）：k 个门控网络（k 为任务数量）。每个门控网络学习为当前输入和对应任务，计算出一个权重分布，来决定各个专家输出的组合比例。

```python
import numpy as np
# 假设我们使用一个深度学习框架（如PyTorch/TensorFlow）的风格来编写
def MMoE(input_x, num_experts, num_tasks, expert_output_dim):
    """
    MMoE 层的前向传播伪代码。

    Args:
        input_x: 输入张量，形状为 (batch_size, input_dim)
        num_experts: 专家数量 (n)
        num_tasks: 任务数量 (k)
        expert_output_dim: 每个专家的输出维度

    Returns:
        task_outputs: 一个列表，包含 k 个任务的最终输出。
                      每个任务的输出形状为 (batch_size, expert_output_dim)
    """

    # 1. 初始化专家网络和门控网络
    # 假设 ‘dense’ 是一个全连接层，‘softmax’ 是softmax激活函数
    experts = [Dense(units=expert_output_dim, activation='relu') 
               for _ in range(num_experts)]
    
    gates = [Dense(units=num_experts, activation='softmax') 
             for _ in range(num_tasks)]

    # 2. 专家前向传播
    # 每个专家处理相同的输入
    expert_outputs = []
    for expert in experts:
        # 每个 expert_out 的形状: (batch_size, expert_output_dim)
        expert_out = expert(input_x)
        expert_outputs.append(expert_out)

    # 将专家输出堆叠起来，便于后续计算
    # expert_outputs_stack 的形状: (batch_size, num_experts, expert_output_dim)
    expert_outputs_stack = stack(expert_outputs, axis=1)

    # 3. 门控网络前向传播与最终输出计算
    task_outputs = [] # 用于存储每个任务的最终输出

    for i in range(num_tasks): # 遍历每个任务
        # 获取当前任务的门控网络
        gate_network = gates[i] 
        # gate_weights 的形状: (batch_size, num_experts)
        # 每一行是一个概率分布，代表每个专家对于当前样本和当前任务的重要性
        gate_weights = gate_network(input_x) 

        # 4. 为每个样本进行加权求和
        # 我们需要扩展 gate_weights 的维度以进行矩阵乘法
        # gate_weights_expanded 的形状: (batch_size, num_experts, 1)
        gate_weights_expanded = expand_dims(gate_weights, axis=-1)

        # 5. 计算加权和
        # 广播机制：gate_weights_expanded * expert_outputs_stack
        # 结果的形状: (batch_size, num_experts, expert_output_dim)
        # 然后在 ‘专家’ 维度（axis=1）上进行求和，得到最终输出
        task_output = sum(gate_weights_expanded * expert_outputs_stack, axis=1)
        # task_output 的形状: (batch_size, expert_output_dim)

        task_outputs.append(task_output)

    return task_outputs

# 示例用法
# batch_size = 32, input_dim = 100, n=5 experts, k=2 tasks, expert_output_dim=64
input_tensor = random((32, 100))
outputs = MMoE(input_tensor, num_experts=5, num_tasks=2, expert_output_dim=64)
```

* MLA
独特设计：
* 潜在序列 (latent_k, latent_v): 这是固定长度的序列（例如 256），用于替代不断增长的原始 KV 序列。
* 投影器 (to_latent_k, to_latent_v): 将当前输入的 K 和 V 投影到与潜在空间兼容的表示。这里的实现是简单的线性层，原论文可能使用了更精细的设计。
* 更新机制: 我们使用一个滑动窗口策略来更新潜在序列。将新的投影后的 KV 与旧的潜在序列拼接，然后只保留最后 latent_len 个向量。这是一种直观的 FIFO（先进先出）队列方式。
* 注意力计算: Query 与整个潜在序列（而不是整个历史序列）计算注意力分数，然后用这些分数对潜在序列中的 Value 进行加权求和。
* 内存优势: 无论处理多长的序列，kv_cache 的大小始终是固定的 [batch, n_heads, latent_len, head_dim]，实现了 O(1) 的内存复杂度，这是 MLA 的核心优势。
## 代码实现
```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, einsum

class MultiHeadLatentAttention(nn.Module):
    """
    简化版的 Multi-Head Latent Attention (MLA) 模块。
    
    Args:
        dim (int): 输入特征的维度（model dimension）。
        n_heads (int): 注意力头的数量。
        head_dim (int): 每个注意力头的维度。
        latent_len (int): 潜在序列的固定长度 (M)。
        scale (float, optional): 缩放因子，通常为 head_dim ** -0.5。
    """
    def __init__(self, dim, n_heads, head_dim, latent_len=512, scale=None):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.latent_len = latent_len
        self.scale = scale or head_dim ** -0.5

        # 投影矩阵：将输入映射到 Q, K, V
        self.to_q = nn.Linear(dim, n_heads * head_dim, bias=False)
        self.to_k = nn.Linear(dim, n_heads * head_dim, bias=False)
        self.to_v = nn.Linear(dim, n_heads * head_dim, bias=False)
        
        # 输出投影
        self.to_out = nn.Linear(n_heads * head_dim, dim)

        # MLA 核心：将原始 K, V 序列投影到潜在空间的投影器
        # 这里使用简单的线性层。原论文可能更复杂。
        self.to_latent_k = nn.Linear(head_dim, head_dim, bias=False)
        self.to_latent_v = nn.Linear(head_dim, head_dim, bias=False)
        
        # 可学习的潜在向量，用于初始化潜在序列
        self.latent_k = nn.Parameter(torch.randn(latent_len, head_dim))
        self.latent_v = nn.Parameter(torch.randn(latent_len, head_dim))

        # 初始化潜在向量
        nn.init.normal_(self.latent_k, std=0.02)
        nn.init.normal_(self.latent_v, std=0.02)

    def forward(self, x, past_kv=None):
        """
        Args:
            x (torch.Tensor): 输入序列，形状为 [batch_size, seq_len, dim]。
            past_kv (tuple, optional): 之前的潜在 KV 状态，形状为 
                ([batch, n_heads, latent_len, head_dim], [batch, n_heads, latent_len, head_dim])。
                如果是第一次推理，则为 None。

        Returns:
            out (torch.Tensor): 注意力输出，形状为 [batch_size, seq_len, dim]。
            new_kv (tuple): 更新后的潜在 KV 状态，用于下一次推理。
        """
        batch_size, seq_len, _ = x.shape
        
        # 1. 投影得到 Q, K, V
        q = self.to_q(x)  # [batch, seq_len, n_heads * head_dim]
        k = self.to_k(x)  # [batch, seq_len, n_heads * head_dim]
        v = self.to_v(x)  # [batch, seq_len, n_heads * head_dim]

        # 重排为多头形式
        q = rearrange(q, 'b s (h d) -> b h s d', h=self.n_heads)
        k = rearrange(k, 'b s (h d) -> b h s d', h=self.n_heads)
        v = rearrange(v, 'b s (h d) -> b h s d', h=self.n_heads)

        # 2. 处理 KV 的潜在序列
        if past_kv is None:
            # 第一次推理，初始化潜在序列
            latent_k = self.latent_k.unsqueeze(0).unsqueeze(0)  # [1, 1, latent_len, head_dim]
            latent_v = self.latent_v.unsqueeze(0).unsqueeze(0)  # [1, 1, latent_len, head_dim]
            # 广播到 batch 和 n_heads
            latent_k = latent_k.expand(batch_size, self.n_heads, -1, -1)
            latent_v = latent_v.expand(batch_size, self.n_heads, -1, -1)
        else:
            # 使用之前传递过来的潜在 KV
            latent_k, latent_v = past_kv

        # 3. 将当前步的 K, V 投影到潜在空间
        # 这里使用线性投影，原论文可能使用更复杂的机制（如跨头共享、更复杂的变换）
        proj_k = self.to_latent_k(k)  # [batch, n_heads, seq_len, head_dim]
        proj_v = self.to_latent_v(v)  # [batch, n_heads, seq_len, head_dim]

        # 4. 更新潜在序列 (类似 FIFO 队列)
        # 我们用当前步的新 KV 替换掉潜在序列中最老的部分（滑动窗口）
        # 这里是一种简化策略：直接拼接并取最后 latent_len 个
        combined_k = torch.cat([latent_k, proj_k], dim=2)
        combined_v = torch.cat([latent_v, proj_v], dim=2)
        
        # 保持潜在序列长度固定为 self.latent_len
        new_latent_k = combined_k[:, :, -self.latent_len :, :]
        new_latent_v = combined_v[:, :, -self.latent_len :, :]

        # 5. 计算注意力
        # Query 与潜在序列中的 Key 计算注意力
        attn_scores = einsum(q, new_latent_k, "b h s d, b h l d -> b h s l") * self.scale
        attn_weights = F.softmax(attn_scores, dim=-1)
        
        # 用注意力权重加权潜在序列中的 Value
        out = einsum(attn_weights, new_latent_v, "b h s l, b h l d -> b h s d")
        
        # 6. 合并多头输出并投影
        out = rearrange(out, 'b h s d -> b s (h d)')
        out = self.to_out(out)

        # 返回输出和更新后的潜在 KV 状态（用于下一个推理步）
        return out, (new_latent_k.detach(), new_latent_v.detach())

```