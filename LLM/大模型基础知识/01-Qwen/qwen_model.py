import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class RMSNorm(nn.Module):
    """RMSNorm实现"""
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight

def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    """预计算RoPE的频率和复数形式"""
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # 复数形式 e^(i*theta)
    return freqs_cis

def apply_rope(xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor):
    """应用RoPE位置编码"""
    xq_ = xq.float().reshape(*xq.shape[:-1], -1, 2)
    xk_ = xk.float().reshape(*xk.shape[:-1], -1, 2)
    
    xq_ = torch.view_as_complex(xq_)
    xk_ = torch.view_as_complex(xk_)
    
    freqs_cis = freqs_cis.view(1, xq_.size(1), 1)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(2)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(2)
    
    return xq_out.type_as(xq), xk_out.type_as(xk)

class GroupedQueryAttention(nn.Module):
    """分组查询注意力(GQA)实现"""
    def __init__(self, dim: int, n_heads: int, n_kv_heads: int, head_dim: int):
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        
        # 查询投影
        self.wq = nn.Linear(dim, n_heads * head_dim, bias=False)
        # 键投影 (共享的KV头)
        self.wk = nn.Linear(dim, n_kv_heads * head_dim, bias=False)
        # 值投影 (共享的KV头)
        self.wv = nn.Linear(dim, n_kv_heads * head_dim, bias=False)
        # 输出投影
        self.wo = nn.Linear(n_heads * head_dim, dim, bias=False)
        
        # 缓存用于推理的KV
        self.cache_k = None
        self.cache_v = None

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor, mask: torch.Tensor = None, start_pos: int = 0):
        batch_size, seq_len, _ = x.shape
        
        # 投影得到Q, K, V
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
        
        # 重塑为多头形式
        xq = xq.view(batch_size, seq_len, self.n_heads, self.head_dim)
        xk = xk.view(batch_size, seq_len, self.n_kv_heads, self.head_dim)
        xv = xv.view(batch_size, seq_len, self.n_kv_heads, self.head_dim)
        
        # 应用RoPE位置编码
        xq, xk = apply_rope(xq, xk, freqs_cis)
        
        # 替换维度以便于矩阵乘法
        xq = xq.transpose(1, 2)  # [bs, n_heads, seq_len, head_dim]
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)
        
        # 处理KV缓存（用于推理时的自回归生成）
        if self.cache_k is not None:
            xk = torch.cat([self.cache_k, xk], dim=2)
            xv = torch.cat([self.cache_v, xv], dim=2)
            self.cache_k = xk
            self.cache_v = xv
        
        # 复制KV头来匹配Q头的数量
        keys = xk.repeat_interleave(self.n_heads // self.n_kv_heads, dim=1)
        values = xv.repeat_interleave(self.n_heads // self.n_kv_heads, dim=1)
        
        # 计算注意力分数
        scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(self.head_dim)
        
        if mask is not None:
            scores = scores + mask
        
        # Softmax得到注意力权重
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)
        
        # 应用注意力权重到values上
        output = torch.matmul(scores, values)
        
        # 转换维度并重塑
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        
        return self.wo(output)

class SwiGLUFFN(nn.Module):
    """SwiGLU激活函数的前馈网络"""
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(dim, hidden_dim, bias=False)
        self.w3 = nn.Linear(hidden_dim, dim, bias=False)
        
    def forward(self, x):
        return self.w3(F.silu(self.w1(x)) * self.w2(x))

class TransformerBlock(nn.Module):
    """单个Transformer解码器块"""
    def __init__(self, dim: int, n_heads: int, n_kv_heads: int, head_dim: int, hidden_dim: int):
        super().__init__()
        self.attention_norm = RMSNorm(dim)
        self.ffn_norm = RMSNorm(dim)
        self.attention = GroupedQueryAttention(dim, n_heads, n_kv_heads, head_dim)
        self.ffn = SwiGLUFFN(dim, hidden_dim)
        
    def forward(self, x, freqs_cis, mask, start_pos):
        # 注意力子层（带残差连接）
        h = x + self.attention(self.attention_norm(x), freqs_cis, mask, start_pos)
        # 前馈子层（带残差连接）
        h = h + self.ffn(self.ffn_norm(h))
        return h

class Qwen2(nn.Module):
    """简化的Qwen2模型"""
    def __init__(self, vocab_size: int, dim: int, n_layers: int, n_heads: int, 
                 n_kv_heads: int, head_dim: int, hidden_dim: int, max_seq_len: int = 2048):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        self.n_layers = n_layers
        
        # 词嵌入
        self.tok_embeddings = nn.Embedding(vocab_size, dim)
        
        # 预计算RoPE频率
        self.freqs_cis = precompute_freqs_cis(head_dim, max_seq_len * 2)
        
        # Transformer层
        self.layers = nn.ModuleList([
            TransformerBlock(dim, n_heads, n_kv_heads, head_dim, hidden_dim)
            for _ in range(n_layers)
        ])
        
        # 输出归一化和线性层
        self.norm = RMSNorm(dim)
        self.output = nn.Linear(dim, vocab_size, bias=False)
        
        # 掩码（用于防止看到未来的token）
        self.register_buffer("mask", torch.tril(torch.ones(max_seq_len, max_seq_len)))
        
    def forward(self, tokens: torch.Tensor, start_pos: int = 0):
        batch_size, seq_len = tokens.shape
        
        # 获取嵌入
        h = self.tok_embeddings(tokens)
        
        # 获取RoPE频率和掩码
        freqs_cis = self.freqs_cis[start_pos: start_pos + seq_len]
        mask = self.mask[:seq_len, :seq_len]
        
        # 通过所有Transformer层
        for layer in self.layers:
            h = layer(h, freqs_cis, mask, start_pos)
        
        # 最终输出
        h = self.norm(h)
        output = self.output(h)
        return output

# 示例用法
if __name__ == "__main__":
    # 定义模型参数（以Qwen2-7B为例）
    vocab_size = 151936  # Qwen2的词表大小
    dim = 4096           # 隐藏层维度
    n_layers = 1        # 层数
    n_heads = 32         # 注意力头数
    n_kv_heads = 8       # KV头数（分组查询）
    head_dim = 128       # 每个头的维度
    hidden_dim = 14336   # FFN隐藏层维度
    
    # 创建模型
    model = Qwen2(vocab_size, dim, n_layers, n_heads, n_kv_heads, head_dim, hidden_dim)
    
    # 示例输入
    tokens = torch.randint(0, vocab_size, (1, 10))  # batch_size=1, seq_len=10
    
    # 前向传播
    with torch.no_grad():
        output = model(tokens)
        print(f"输入形状: {tokens.shape}")
        print(f"输出形状: {output.shape}")  # 应该是 [1, 10, vocab_size]
        print("模型参数量:", sum(p.numel() for p in model.parameters()))