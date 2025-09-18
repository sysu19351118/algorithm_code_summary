

import torch
import pdb
import json
from tqdm import tqdm


def save_sid2item_info(sid_path, save_path):
    data = torch.load(sid_path)
    sid_count = {}
    for sid in tqdm(data['semantic_ids']):
        key = "-".join([str(i) for i in sid.tolist()])
        if key in sid_count:
            sid_count[key]+=1
        else:
            sid_count[key]=1
    # 保存为JSON文件
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(sid_count, f, ensure_ascii=False, indent=4)
    


if __name__ == "__main__":
    sid_path = '/home/tangzixuan.8/Generative_recall/generative_recall_rank/data/5-core/Video_Games/OneRecCache/sid/rqkmeans_semantic_ids.pt'
    save_path = '/home/tangzixuan.8/Generative_recall/generative_recall_rank/data/5-core/Video_Games/OneRecCache/sid/semantic_ids_2_item.json'
    save_sid2item_info(sid_path, save_path)
