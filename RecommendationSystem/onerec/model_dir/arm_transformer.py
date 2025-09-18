
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

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# class PositionalEncoding(nn.Module):
#     def __init__(self, d_model, max_len=5000):
#         super(PositionalEncoding, self).__init__()
#         pe = torch.zeros(max_len, d_model)
#         position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
#         div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
#         pe[:, 0::2] = torch.sin(position * div_term)
#         pe[:, 1::2] = torch.cos(position * div_term)
#         pe = pe.unsqueeze(0)
#         self.register_buffer('pe', pe)

#     def forward(self, x):
#         return x + self.pe[:, :x.size(1)]

# class MultiHeadAttention(nn.Module):
#     def __init__(self, d_model, num_heads):
#         super(MultiHeadAttention, self).__init__()
#         assert d_model % num_heads == 0
#         self.d_model = d_model
#         self.num_heads = num_heads
#         self.d_k = d_model // num_heads
        
#         self.w_q = nn.Linear(d_model, d_model)
#         self.w_k = nn.Linear(d_model, d_model)
#         self.w_v = nn.Linear(d_model, d_model)
#         self.w_o = nn.Linear(d_model, d_model)
        
#     def scaled_dot_product_attention(self, q, k, v, mask=None):
#         attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
#         if mask is not None:
#             attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
#         attn_probs = F.softmax(attn_scores, dim=-1)
#         output = torch.matmul(attn_probs, v)
#         return output
        
#     def split_heads(self, x):
#         batch_size, seq_len, d_model = x.size()
#         return x.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        
#     def combine_heads(self, x):
#         batch_size, _, seq_len, d_k = x.size()
#         return x.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        
#     def forward(self, q, k, v, mask=None):
#         q = self.split_heads(self.w_q(q))
#         k = self.split_heads(self.w_k(k))
#         v = self.split_heads(self.w_v(v))
        
#         attn_output = self.scaled_dot_product_attention(q, k, v, mask)
#         output = self.w_o(self.combine_heads(attn_output))
#         return output

# class FeedForward(nn.Module):
#     def __init__(self, d_model, d_ff):
#         super(FeedForward, self).__init__()
#         self.linear1 = nn.Linear(d_model, d_ff)
#         self.linear2 = nn.Linear(d_ff, d_model)
#         self.dropout = nn.Dropout(0.1)
        
#     def forward(self, x):
#         return self.linear2(self.dropout(F.relu(self.linear1(x))))

# class EncoderLayer(nn.Module):
#     def __init__(self, d_model, num_heads, d_ff):
#         super(EncoderLayer, self).__init__()
#         self.self_attn = MultiHeadAttention(d_model, num_heads)
#         self.feed_forward = FeedForward(d_model, d_ff)
#         self.norm1 = nn.LayerNorm(d_model)
#         self.norm2 = nn.LayerNorm(d_model)
#         self.dropout = nn.Dropout(0.1)
        
#     def forward(self, x, mask=None):
#         attn_output = self.self_attn(x, x, x, mask)
#         x = self.norm1(x + self.dropout(attn_output))
#         ff_output = self.feed_forward(x)
#         x = self.norm2(x + self.dropout(ff_output))
#         return x

# class DecoderLayer(nn.Module):
#     def __init__(self, d_model, num_heads, d_ff):
#         super(DecoderLayer, self).__init__()
#         self.self_attn = MultiHeadAttention(d_model, num_heads)
#         self.cross_attn = MultiHeadAttention(d_model, num_heads)
#         self.feed_forward = FeedForward(d_model, d_ff)
#         self.norm1 = nn.LayerNorm(d_model)
#         self.norm2 = nn.LayerNorm(d_model)
#         self.norm3 = nn.LayerNorm(d_model)
#         self.dropout = nn.Dropout(0.1)
        
#     def forward(self, x, enc_output, src_mask=None, tgt_mask=None):
#         attn_output = self.self_attn(x, x, x, tgt_mask)
#         x = self.norm1(x + self.dropout(attn_output))
        
#         cross_attn_output = self.cross_attn(x, enc_output, enc_output, src_mask)
#         x = self.norm2(x + self.dropout(cross_attn_output))
        
#         ff_output = self.feed_forward(x)
#         x = self.norm3(x + self.dropout(ff_output))
#         return x

# class Transformer(nn.Module):
#     def __init__(self, src_vocab_size, tgt_vocab_size, d_model, num_heads, d_ff, num_layers, max_seq_len):
#         super(Transformer, self).__init__()
#         self.encoder_embedding = nn.Embedding(src_vocab_size, d_model)
#         self.decoder_embedding = nn.Embedding(tgt_vocab_size, d_model)
#         self.positional_encoding = PositionalEncoding(d_model, max_seq_len)
        
#         self.encoder_layers = nn.ModuleList([
#             EncoderLayer(d_model, num_heads, d_ff) for _ in range(num_layers // 2)
#         ])
        
#         self.decoder_layers = nn.ModuleList([
#             DecoderLayer(d_model, num_heads, d_ff) for _ in range(num_layers // 2)
#         ])
        
#         self.fc_out = nn.Linear(d_model, tgt_vocab_size)
#         self.dropout = nn.Dropout(0.1)
        
#     def generate_mask(self, src, tgt):
#         src_mask = (src != 0).unsqueeze(1).unsqueeze(2)
#         tgt_mask = (tgt != 0).unsqueeze(1).unsqueeze(3)
#         seq_len = tgt.size(1)
#         nopeak_mask = (1 - torch.triu(torch.ones(1, seq_len, seq_len), diagonal=1)).bool()
#         tgt_mask = tgt_mask & nopeak_mask
#         return src_mask, tgt_mask
        
#     def forward(self, src, tgt):
#         src_mask, tgt_mask = self.generate_mask(src, tgt)
        
#         # Encoder
#         enc_output = self.dropout(self.positional_encoding(self.encoder_embedding(src)))
#         for layer in self.encoder_layers:
#             enc_output = layer(enc_output, src_mask)
        
#         # Decoder
#         dec_output = self.dropout(self.positional_encoding(self.decoder_embedding(tgt)))
#         for layer in self.decoder_layers:
#             dec_output = layer(dec_output, enc_output, src_mask, tgt_mask)
        
#         output = self.fc_out(dec_output)
#         return output
    
#     def generate(self, src, start_token, max_len, temperature=1.0):
#         self.eval()
#         with torch.no_grad():
#             # Encode source
#             src_mask = (src != 0).unsqueeze(1).unsqueeze(2)
#             enc_output = self.dropout(self.positional_encoding(self.encoder_embedding(src)))
#             for layer in self.encoder_layers:
#                 enc_output = layer(enc_output, src_mask)
            
#             # Initialize target with start token
#             generated = torch.tensor([[start_token]], device=src.device)
            
#             for _ in range(max_len - 1):
#                 # Generate target mask
#                 tgt_mask = (generated != 0).unsqueeze(1).unsqueeze(3)
#                 seq_len = generated.size(1)
#                 nopeak_mask = (1 - torch.triu(torch.ones(1, seq_len, seq_len, device=src.device), diagonal=1)).bool()
#                 tgt_mask = tgt_mask & nopeak_mask
                
#                 # Decode
#                 dec_output = self.dropout(self.positional_encoding(self.decoder_embedding(generated)))
#                 for layer in self.decoder_layers:
#                     dec_output = layer(dec_output, enc_output, src_mask, tgt_mask)
                
#                 # Get next token
#                 logits = self.fc_out(dec_output[:, -1, :]) / temperature
#                 next_token = torch.argmax(logits, dim=-1).unsqueeze(0)
                
#                 # Append to generated sequence
#                 generated = torch.cat([generated, next_token], dim=1)
                
#                 # Stop if end token is generated
#                 if next_token.item() == 0:  # Assuming 0 is padding/end token
#                     break
            
#             return generated

# # Example usage
# if __name__ == "__main__":
#     # Hyperparameters
#     src_vocab_size = 1000
#     tgt_vocab_size = 1000
#     d_model = 512
#     num_heads = 8
#     d_ff = 2048
#     num_layers = 6  # Total layers, encoder will have 3, decoder will have 3
#     max_seq_len = 100
    
#     # Create model
#     model = OneRecModel(src_vocab_size, tgt_vocab_size, d_model, num_heads, d_ff, num_layers, max_seq_len)
    
#     # Example input
#     src = torch.randint(0, src_vocab_size, (1, 10))  # (batch_size, seq_len)
#     tgt = torch.randint(0, tgt_vocab_size, (1, 5))
    
#     # Forward pass
#     output = model(src, tgt)
#     print("Output shape:", output.shape)  # Should be (1, 5, tgt_vocab_size)
    
#     # Generate sequence
#     generated = model.generate(src, start_token=1, max_len=20)
#     print("Generated sequence:", generated)






import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import List, Optional

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
    def scaled_dot_product_attention(self, q, k, v, mask=None):
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
        attn_probs = F.softmax(attn_scores, dim=-1)
        output = torch.matmul(attn_probs, v)
        return output
        
    def split_heads(self, x):
        batch_size, seq_len, d_model = x.size()
        return x.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        
    def combine_heads(self, x):
        batch_size, _, seq_len, d_k = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        
    def forward(self, q, k, v, mask=None):
        q = self.split_heads(self.w_q(q))
        k = self.split_heads(self.w_k(k))
        v = self.split_heads(self.w_v(v))
        
        attn_output = self.scaled_dot_product_attention(q, k, v, mask)
        output = self.w_o(self.combine_heads(attn_output))
        return output

class Expert(nn.Module):
    """单个专家网络"""
    def __init__(self, d_model, d_ff):
        super(Expert, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x):
        return self.linear2(self.dropout(F.gelu(self.linear1(x))))

class MoELayer(nn.Module):
    """Mixture of Experts层"""
    def __init__(self, d_model, d_ff, num_experts, top_k=2, expert_capacity_factor=1.0):
        super(MoELayer, self).__init__()
        self.d_model = d_model
        self.num_experts = num_experts
        self.top_k = top_k
        self.expert_capacity_factor = expert_capacity_factor
        
        # 创建多个专家
        self.experts = nn.ModuleList([Expert(d_model, d_ff) for _ in range(num_experts)])
        
        # 门控网络
        self.gate = nn.Linear(d_model, num_experts, bias=False)
        
        # 辅助损失相关
        self.aux_loss = 0.0
        
    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        
        # 计算门控权重
        gate_logits = self.gate(x)  # [batch_size, seq_len, num_experts]
        gate_probs = F.softmax(gate_logits, dim=-1)
        
        # 选择top-k专家
        top_k_gate_probs, top_k_indices = torch.topk(gate_probs, self.top_k, dim=-1)
        top_k_gate_probs = top_k_gate_probs / top_k_gate_probs.sum(dim=-1, keepdim=True)
        
        # 计算辅助损失（负载均衡损失）
        self.compute_aux_loss(gate_probs, top_k_indices)
        
        # 初始化输出
        output = torch.zeros_like(x)
        
        # 为每个专家处理输入
        for expert_idx in range(self.num_experts):
            # 找出当前专家需要处理的token
            expert_mask = (top_k_indices == expert_idx)
            expert_weights = top_k_gate_probs * expert_mask.float()
            
            # 重新组织输入
            flat_expert_mask = expert_mask.reshape(-1, self.top_k)
            flat_input = x.reshape(-1, self.d_model)
            
            # 选择当前专家需要处理的token
            selected_indices = flat_expert_mask.any(dim=1)
            if selected_indices.sum() > 0:
                expert_input = flat_input[selected_indices]
                
                # 专家前向传播
                expert_output = self.experts[expert_idx](expert_input)
                
                # 计算加权输出
                flat_weights = expert_weights.reshape(-1, self.top_k).sum(dim=1, keepdim=True)[selected_indices]
                weighted_output = expert_output * flat_weights
                
                # 将输出放回正确位置
                output_flat = output.reshape(-1, self.d_model)
                output_flat[selected_indices] += weighted_output
                output = output_flat.reshape(batch_size, seq_len, self.d_model)
        
        return output
    
    def compute_aux_loss(self, gate_probs, top_k_indices):
        """计算负载均衡辅助损失"""
        batch_size, seq_len, _ = gate_probs.shape
        
        # 计算每个专家的选择频率
        expert_mask = F.one_hot(top_k_indices, self.num_experts).float()
        expert_usage = expert_mask.sum(dim=1).sum(dim=0)  # [num_experts]
        
        # 计算每个专家的门控概率总和
        gate_sum = gate_probs.sum(dim=0).sum(dim=0)  # [num_experts]
        
        # 计算负载均衡损失
        expert_usage_rate = expert_usage / (batch_size * seq_len * self.top_k)
        gate_sum_rate = gate_sum / (batch_size * seq_len)
        
        # 方差损失，鼓励均匀分布
        aux_loss = torch.std(expert_usage_rate) * torch.std(gate_sum_rate)
        self.aux_loss = aux_loss
        
        return aux_loss

class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = FeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x, mask=None):
        attn_output = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x

class MoEDecoderLayer(nn.Module):
    """使用MoE的Decoder层"""
    def __init__(self, d_model, num_heads, d_ff, num_experts=4, top_k=2):
        super(MoEDecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.cross_attn = MultiHeadAttention(d_model, num_heads)
        
        # 使用MoE代替普通的FFN
        self.moe_layer = MoELayer(d_model, d_ff, num_experts, top_k)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x, enc_output, src_mask=None, tgt_mask=None):
        # Self attention
        attn_output = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Cross attention
        cross_attn_output = self.cross_attn(x, enc_output, enc_output, src_mask)
        x = self.norm2(x + self.dropout(cross_attn_output))
        
        # MoE FFN
        moe_output = self.moe_layer(x)
        x = self.norm3(x + self.dropout(moe_output))
        
        return x
    
    def get_aux_loss(self):
        """获取MoE的辅助损失"""
        return self.moe_layer.aux_loss

class FeedForward(nn.Module):
    """普通的FFN（用于Encoder）"""
    def __init__(self, d_model, d_ff):
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x):
        return self.linear2(self.dropout(F.gelu(self.linear1(x))))

class OneRecModel(nn.Module):
    def __init__(self, config, num_experts=4, top_k=2):
        super(OneRecModel, self).__init__()
        src_vocab_size = config['model']['src_vocab_size']
        tgt_vocab_size = config['model']['tgt_vocab_size']
        d_model = config['model']['d_model']
        num_heads = config['model']['num_heads']
        d_ff = config['model']['d_ff']
        num_layers = config['model']['num_layers']
        max_seq_len = config['model']['max_seq_len']
        
        self.encoder_embedding = nn.Embedding(src_vocab_size, d_model)
        self.decoder_embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_len)
        self.PAD_TOKEN = config['model']['special_token']['PAD']
        # Encoder使用普通层
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff) for _ in range(num_layers // 2)
        ])
        
        # Decoder使用MoE层
        self.decoder_layers = nn.ModuleList([
            MoEDecoderLayer(d_model, num_heads, d_ff, num_experts, top_k) 
            for _ in range(num_layers // 2)
        ])
        
        self.fc_out = nn.Linear(d_model, tgt_vocab_size)
        self.dropout = nn.Dropout(0.1)
        self.num_experts = num_experts
        self.top_k = top_k
        
    def generate_mask(self, src, tgt):
        src_mask = (src != self.PAD_TOKEN).unsqueeze(1).unsqueeze(2)
        tgt_mask = (tgt != self.PAD_TOKEN).unsqueeze(1).unsqueeze(3)
        seq_len = tgt.size(1)
        nopeak_mask = (1 - torch.triu(torch.ones(1, seq_len, seq_len, device=src.device), diagonal=1)).bool()
        tgt_mask = tgt_mask & nopeak_mask
        return src_mask, tgt_mask
        
    def forward(self, src, tgt):
        src_mask, tgt_mask = self.generate_mask(src, tgt)
        # Encoder
        enc_output = self.dropout(self.positional_encoding(self.encoder_embedding(src)))

        for layer in self.encoder_layers:
            enc_output = layer(enc_output, src_mask)

        # Decoder
        dec_output = self.dropout(self.positional_encoding(self.decoder_embedding(tgt)))
        total_aux_loss = 0.0
        for layer in self.decoder_layers:
            dec_output = layer(dec_output, enc_output, src_mask, tgt_mask)
            total_aux_loss += layer.get_aux_loss()
        
        output = self.fc_out(dec_output)
        
        # 返回输出和辅助损失
        return output, total_aux_loss
    
    def generate(self, src, start_token, max_len, temperature=1.0, beam_size=5):
        self.eval()
        with torch.no_grad():
            # 编码器部分保持不变
            src_mask = (src != self.PAD_TOKEN).unsqueeze(1).unsqueeze(2)
            enc_output = self.dropout(self.positional_encoding(self.encoder_embedding(src)))
            for layer in self.encoder_layers:
                enc_output = layer(enc_output, src_mask)
            
            # 初始化beam search
            beams = [{
                'sequence': torch.tensor([[start_token]], device=src.device),
                'score': 0.0,
                'ended': False
            }]
            
            final_results = []
            
            for step in range(max_len - 1):
                # 如果所有beam都已结束或没有候选，提前终止
                if all(beam['ended'] for beam in beams) and len(beams) > 0:
                    break
                    
                # 收集所有需要扩展的beam
                candidates = []
                for beam in beams:
                    if beam['ended']:
                        # 已结束的beam直接加入候选
                        candidates.append(beam)
                        continue
                    
                    # 准备解码器输入
                    generated = beam['sequence']
                    tgt_mask = (generated != 0).unsqueeze(1).unsqueeze(3)
                    seq_len = generated.size(1)
                    nopeak_mask = (1 - torch.triu(torch.ones(1, seq_len, seq_len, device=src.device), diagonal=1)).bool()
                    tgt_mask = tgt_mask & nopeak_mask
                    
                    # 解码器前向传播
                    dec_output = self.dropout(self.positional_encoding(self.decoder_embedding(generated)))
                    for layer in self.decoder_layers:
                        dec_output = layer(dec_output, enc_output, src_mask, tgt_mask)
                    
                    # 获取下一个token的概率分布
                    logits = self.fc_out(dec_output[:, -1, :]) / temperature
                    probs = torch.softmax(logits, dim=-1)
                    
                    # 获取top-k候选
                    topk_probs, topk_indices = torch.topk(probs, beam_size, dim=-1)
                    
                    # 为当前beam生成新的候选
                    for i in range(beam_size):
                        next_token = topk_indices[0, i].item()
                        next_prob = topk_probs[0, i].item()
                        
                        new_sequence = torch.cat([generated, torch.tensor([[next_token]], device=src.device)], dim=1)
                        new_score = beam['score'] - torch.log(torch.tensor(next_prob)).item()  # 负对数似然
                        
                        candidate = {
                            'sequence': new_sequence,
                            'score': new_score,
                            'ended': (next_token == 0)  # 遇到结束符
                        }
                        candidates.append(candidate)
                
                # 按分数排序并选择top beam_size个候选
                candidates.sort(key=lambda x: x['score'])
                beams = candidates[:beam_size]
                
                # 将已结束的beam移到最终结果中
                new_beams = []
                for beam in beams:
                    if beam['ended']:
                        final_results.append(beam)
                    else:
                        new_beams.append(beam)
                beams = new_beams
            
            # 将剩余的beam加入最终结果
            final_results.extend(beams)
            
            # 按分数排序最终结果
            final_results.sort(key=lambda x: x['score'])
            
            # 返回所有生成的序列（去掉开始token和结束token）
            return [result['sequence'][0].tolist() for result in final_results[:beam_size]]


def load_config(config_path):
    """加载YAML配置文件"""
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

# 示例用法
if __name__ == "__main__":
    # 超参数
    config = load_config('/home/tangzixuan.8/Generative_recall/generative_recall_rank/compare_exp/onerec/config/toys_onerec.yaml') 
    src_vocab_size = 200
    tgt_vocab_size = 200
    d_model = 512
    num_heads = 8
    d_ff = 2048
    num_layers = 6
    max_seq_len = 100
    num_experts = 4
    top_k = 2
    
    # 创建MoE Transformer
    model = OneRecModel(
        config
    )
    
    # 示例输入
    src = torch.randint(0, src_vocab_size, (8, 220))
    tgt = torch.randint(0, tgt_vocab_size, (8, 4))
    
    # # 前向传播
    # output, aux_loss = model(src, tgt)
    # pdb.set_trace()
    # print("Output shape:", output.shape)
    # print("Auxiliary loss:", aux_loss.item())
    
    # 生成示例
    generated = model.generate(src, start_token=1, max_len=4)
    print("Generated sequence:", generated)
