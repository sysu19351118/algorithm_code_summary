import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from datasets import load_dataset
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
from trl.core import LengthSampler

# 1. 加载模型和tokenizer
model_name = "Qwen/Qwen2-7B"  # 根据实际情况选择Qwen2的版本

# 加载基础模型
model = AutoModelForCausalLMWithValueHead.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)

# 加载tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token  # 设置pad token

# 2. 初始化PPO配置
config = PPOConfig(
    batch_size=8,
    mini_batch_size=4,
    learning_rate=1.41e-5,
    log_with="wandb",  # 可以使用tensorboard或其他
    ppo_epochs=4,
    seed=42,
    steps=10000,
    init_kl_coef=0.2,
    adap_kl_ctrl=True,
    target_kl=0.1,
    gradient_accumulation_steps=1,
)

# 3. 准备数据集
dataset = load_dataset("imdb", split="train")  # 示例数据集，实际使用时替换为你的RLHF数据集
dataset = dataset.rename_columns({"text": "review"})
dataset = dataset.filter(lambda x: len(x["review"]) > 200)  # 过滤短文本

# 4. 定义生成配置
gen_kwargs = {
    "min_length": -1,
    "top_k": 0.0,
    "top_p": 1.0,
    "do_sample": True,
    "pad_token_id": tokenizer.eos_token_id,
    "max_new_tokens": 40,  # 控制生成长度
}

# 5. 定义奖励函数 - 这需要根据你的具体任务定制
def reward_fn(responses):
    """计算生成的响应获得的奖励"""
    rewards = []
    for response in responses:
        # 这里只是一个示例 - 实际使用时需要根据你的任务设计奖励函数
        # 可以是基于规则、基于模型评分或其他方法
        score = len(response) / 100  # 示例: 奖励生成长度
        rewards.append(torch.tensor(score))
    return rewards

# 6. 初始化PPO训练器
ppo_trainer = PPOTrainer(
    config=config,
    model=model,
    ref_model=None,  # 可以设置为原始模型作为参考
    tokenizer=tokenizer,
    dataset=dataset,
)

# 7. 训练循环
for epoch, batch in enumerate(ppo_trainer.dataloader):
    if epoch >= config.total_ppo_epochs:
        break
        
    # 获取查询文本
    query_tensors = [tokenizer.encode(q, return_tensors="pt").squeeze() for q in batch["review"]]
    
    # 生成响应
    response_tensors = []
    for query in query_tensors:
        response = ppo_trainer.generate(
            query.unsqueeze(dim=0).to(model.device),
            **gen_kwargs
        )
        response_tensors.append(response.squeeze())
    
    # 解码文本用于计算奖励
    batch["response"] = [tokenizer.decode(r.squeeze()) for r in response_tensors]
    
    # 计算奖励
    rewards = reward_fn(batch["response"])
    
    # 运行PPO步骤
    stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
    ppo_trainer.log_stats(stats, batch, rewards)
    
    print(f"Epoch {epoch} | Reward: {torch.mean(torch.stack(rewards)):.4f}")

# 8. 保存训练后的模型
model.save_pretrained("qwen2_ppo_model")
tokenizer.save_pretrained("qwen2_ppo_model")