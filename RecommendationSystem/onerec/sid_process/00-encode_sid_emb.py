import torch
import pandas as pd
import json
from tqdm import tqdm
from transformers import BertTokenizer, BertModel
from typing import List, Dict
import pdb
def encode_item_title(file_path, max_items=None):
    """
    从JSONL文件中读取商品数据
    
    Args:
        file_path: JSONL文件路径
        max_items: 最大处理商品数量（None表示处理所有）
    """
    items = []
    count = 0
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="读取商品数据"):
            if max_items is not None and count >= max_items:
                break
            try:
                data = json.loads(line.strip())
                if 'asin' in data and 'title' in data:
                    items.append({
                        "asin": data['asin'],
                        "title": data['title']
                    })
                    count += 1
            except json.JSONDecodeError:
                print(f"跳过无法解析的行: {line[:100]}...")
                continue
    
    item_df = pd.DataFrame(items)
    print(f"成功读取 {len(item_df)} 个商品")
    return item_df

def bert_encode_titles(item_df, output_path, model_name='/home/tangzixuan.8/Generative_recall/bert-base-uncased', 
                      pooling_strategy='cls', batch_size=32, max_length=128):
    """
    使用BERT对商品标题进行编码并保存
    
    Args:
        item_df: 包含asin和title的DataFrame
        output_path: 输出文件路径
        model_name: BERT模型名称
        pooling_strategy: 池化策略 ('cls', 'mean', 'max')
        batch_size: 批处理大小
        max_length: 最大序列长度
    """
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 加载BERT模型和tokenizer
    print(f"加载模型: {model_name}")
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertModel.from_pretrained(model_name)
    model.to(device)
    model.eval()
    
    # 准备数据
    asins = item_df['asin'].values
    titles = item_df['title'].fillna('').astype(str).tolist()
    
    # 分批处理
    all_embeddings = []
    asin_list = []
    
    print("开始BERT编码...")
    for i in tqdm(range(0, len(titles), batch_size), desc="编码进度"):
        batch_titles = titles[i:i+batch_size]
        batch_asins = asins[i:i+batch_size]
        
        # 过滤空标题
        valid_indices = [idx for idx, title in enumerate(batch_titles) if title.strip()]
        if not valid_indices:
            continue
            
        valid_titles = [batch_titles[idx] for idx in valid_indices]
        valid_asins = [batch_asins[idx] for idx in valid_indices]
        
        # Tokenize
        inputs = tokenizer(
            valid_titles,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        )
        
        # 移动到设备
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # 前向传播
        with torch.no_grad():
            outputs = model(**inputs)
        
        # 选择池化策略
        if pooling_strategy == 'cls':
            embeddings = outputs.last_hidden_state[:, 0, :]  # [CLS] token
        elif pooling_strategy == 'mean':
            # 忽略padding tokens的平均池化
            attention_mask = inputs['attention_mask']
            token_embeddings = outputs.last_hidden_state
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
            sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            embeddings = sum_embeddings / sum_mask
        elif pooling_strategy == 'max':
            attention_mask = inputs['attention_mask']
            token_embeddings = outputs.last_hidden_state
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            token_embeddings[input_mask_expanded == 0] = -1e9  # 将padding设置为很小的值
            embeddings = torch.max(token_embeddings, 1)[0]
        else:
            raise ValueError(f"不支持的pooling策略: {pooling_strategy}")
        
        all_embeddings.append(embeddings.cpu())
        asin_list.extend(valid_asins)
    
    if not all_embeddings:
        raise ValueError("没有生成任何嵌入，请检查数据")
    
    # 合并所有嵌入
    all_embeddings = torch.cat(all_embeddings, dim=0)
    asin_tensor = torch.tensor([int(asin) if asin.isdigit() else hash(asin) % (2**31) for asin in asin_list], 
                              dtype=torch.long)
    
    # 保存为pt文件
    torch.save({
        'asins': asin_tensor,
        'embeddings': all_embeddings,
        'model_name': model_name,
        'pooling_strategy': pooling_strategy,
        'original_asins': asin_list  # 保存原始ASIN用于参考
    }, output_path)
    
    print(f"编码完成！")
    print(f"处理商品数量: {len(asin_list)}")
    print(f"嵌入维度: {all_embeddings.shape}")
    print(f"文件已保存至: {output_path}")

def process_complete_pipeline(input_file, output_file, max_items=None):
    """
    完整的处理流水线
    """
    # 1. 读取数据
    print("步骤1: 读取商品数据")
    item_df = encode_item_title(input_file, max_items)
    # 2. BERT编码
    print("\n步骤2: BERT编码")
    bert_encode_titles(
        item_df=item_df,
        output_path=output_file,
        model_name='/home/tangzixuan.8/Generative_recall/bert-base-uncased',
        pooling_strategy='mean',  # 推荐使用mean pooling
        batch_size=32,
        max_length=128
    )
    
    return item_df

# 使用示例
if __name__ == "__main__":
    # 完整处理流程
    input_file = "/home/tangzixuan.8/Generative_recall/generative_recall_rank/data/5-core/Arts_Crafts_and_Sewing/meta_Arts_Crafts_and_Sewing.json"  # 替换为你的文件路径
    output_file = "/home/tangzixuan.8/Generative_recall/generative_recall_rank/data/5-core/Arts_Crafts_and_Sewing/OneRecCache/sid/videogame_item_embeddings.pt"
    
    # 处理前1000个商品（测试用）
    process_complete_pipeline(input_file, output_file, max_items=None)
    
    # 或者处理所有商品
    # process_complete_pipeline(input_file, output_file)
    
    # 加载保存的嵌入
    def load_embeddings(file_path):
        data = torch.load(file_path)
        print(f"模型: {data['model_name']}")
        print(f"池化策略: {data['pooling_strategy']}")
        print(f"嵌入形状: {data['embeddings'].shape}")
        return data['asins'], data['embeddings'], data['original_asins']
    
    # 加载检查
    # asins, embeddings, original_asins = load_embeddings(output_file)