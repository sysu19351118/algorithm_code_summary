import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pdb

# 1. 定义合成数据集（Toy Example）
class PreferenceDataset(Dataset):
    def __init__(self, tokenizer, num_samples=100):
        self.prompts = [
            "Explain the moon landing",
            "What is photosynthesis?",
            "How does a rocket work?",
            "Define quantum computing"
        ]
        # 生成合成偏好数据：chosen 是合理回答，rejected 是随机错误回答
        self.chosen = [
            "The moon landing in 1969 was a historic event...",
            "Photosynthesis is the process by which plants convert sunlight...",
            "Rockets work by expelling propellant at high speed...",
            "Quantum computing uses qubits to perform calculations..."
        ]
        self.rejected = [
            "The moon is made of cheese.",
            "Photosynthesis is a type of dance.",
            "Rockets are powered by magic.",
            "Quantum computing is a breakfast cereal."
        ]
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        return {
            "prompt": self.prompts[idx],
            "chosen": self.chosen[idx],
            "rejected": self.rejected[idx]
        }

# 2. 加载模型和分词器
model_name = "/mnt/sda1/LLM_model/Qwen/Qwen2___5-0___5B-Instruct"  # 替换为实际模型路径
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
ref_model = AutoModelForCausalLM.from_pretrained(model_name)  # 参考模型（固定）

# 冻结参考模型
for param in ref_model.parameters():
    param.requires_grad_(False)

# 3. 计算序列对数概率（支持批量）
def get_batch_logprobs(model, prompts, responses, device="cuda"):
    # Tokenize 输入（prompt + response）
    batch_texts = [p + r for p, r in zip(prompts, responses)]
    inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True).to(device)
    
    # 获取 response 部分的 mask（忽略 prompt）
    prompt_lens = [len(tokenizer(p, return_tensors="pt")["input_ids"][0]) for p in prompts]
    response_mask = torch.zeros_like(inputs["input_ids"])
    for i, (p_len, r_len) in enumerate(zip(prompt_lens, [len(tokenizer(r)["input_ids"]) for r in responses])):
        response_mask[i, p_len:p_len + r_len] = 1
    
    # 前向计算
    with torch.set_grad_enabled(model.training):
        outputs = model(**inputs)
        logits = outputs.logits
    
    # 计算 response 的对数概率
    logprobs = F.log_softmax(logits, dim=-1)
    token_logprobs = torch.gather(logprobs[:, :-1], -1, inputs["input_ids"][:, 1:].unsqueeze(-1)).squeeze(-1)
    response_logprobs = (token_logprobs * response_mask[:, 1:]).sum(dim=-1)
    

    pdb.set_trace()
    return response_logprobs

# 4. DPO 损失函数
def dpo_loss(model, ref_model, batch, beta=0.1, device="cuda"):
    prompts = batch["prompt"]
    chosen = batch["chosen"]
    rejected = batch["rejected"]
    
    # 计算策略模型的对数概率
    pi_theta_chosen = get_batch_logprobs(model, prompts, chosen, device)
    pi_theta_rejected = get_batch_logprobs(model, prompts, rejected, device)
    
    # 计算参考模型的对数概率
    with torch.no_grad():
        pi_ref_chosen = get_batch_logprobs(ref_model, prompts, chosen, device)
        pi_ref_rejected = get_batch_logprobs(ref_model, prompts, rejected, device)
    
    # 计算对数比值
    log_ratio_chosen = pi_theta_chosen - pi_ref_chosen
    log_ratio_rejected = pi_theta_rejected - pi_ref_rejected
    
    # DPO Loss
    losses = -F.logsigmoid(beta * (log_ratio_chosen - log_ratio_rejected))
    return losses.mean()

# 5. 训练循环
def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    ref_model.to(device)
    
    dataset = PreferenceDataset(tokenizer)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    
    for epoch in range(3):  # 示例：3个epoch
        model.train()
        total_loss = 0
        for batch in dataloader:
            optimizer.zero_grad()
            loss = dpo_loss(model, ref_model, batch, device=device)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        print(f"Epoch {epoch}, Loss: {total_loss / len(dataloader)}")

# 运行训练
train()