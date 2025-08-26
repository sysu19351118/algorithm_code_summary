# AIGC🎬 ｜ LLM💬 ｜ RecSys📖
本仓库为个人整理总结，以AIGC、LLM、搜广推三部分展开。 \
✨ AIGC - 复现了部分领域内的热门算法，使用开源数据集对算法进行复现，提高对算法的理解及应用能力。 \
✨ LLM - 对大模型代码进行精读、对强化学习部分原理进行推导 \
✨ 搜广推 - 对工业界成熟的搜索广告推荐流程进行学习，并结合开源数据集对热门代码进行复现 

## AIGC
使用miniset对AIGC领域的基石论文进行了复现，帮助通透理解这些论文。
* 生成模型经典论文
- [x] [用miniset复现VAE](https://github.com/sysu19351118/algorithm_code_summary/tree/master/AIGC/01-VAE)
- [x] [用miniset复现VQ-VAE](https://github.com/sysu19351118/algorithm_code_summary/tree/master/AIGC/02-VQ-VAE) | [Neural Discrete Representation Learning](https://arxiv.org/pdf/1711.00937)
- [x] [用miniset复现Diffusion (DDPM)](https://github.com/sysu19351118/algorithm_code_summary/tree/master/AIGC/03-DiffusionModel)  | [Denoising diffusion probabilistic models](https://proceedings.neurips.cc/paper_files/paper/2020/file/4c5bcfec8584af0d967f1ab10179ca4b-Paper.pdf)
- [x] [Laten Diffusion (LDM)](https://github.com/sysu19351118/algorithm_code_summary/tree/master/AIGC/04-LatentDiffusion) | [High-Resolution Image Synthesis with Latent Diffusion Models](https://arxiv.org/pdf/2112.10752)
- [x] [Diffusion Transformer (DiT)](https://github.com/sysu19351118/algorithm_code_summary/tree/master/AIGC/05-DiT) | [Scalable Diffusion Models with Transformers](https://arxiv.org/pdf/2212.09748)
- [ ] Flow Matching
* diffusion 生成模型推理加速
- [ ] DDIM

## LLM
* 强化学习理论基础 
- [x] [DPO](https://github.com/sysu19351118/algorithm_code_summary/tree/master/LLM/%E5%BC%BA%E5%8C%96%E5%AD%A6%E4%B9%A0/00-dpo)
- [x] [PPO](https://github.com/sysu19351118/algorithm_code_summary/tree/master/LLM/%E5%BC%BA%E5%8C%96%E5%AD%A6%E4%B9%A0/01-ppo)
- [ ] GRPO
* LLM模型知识整理
- [x] [LLaMA技术细节整理](https://github.com/sysu19351118/algorithm_code_summary/tree/master/LLM/%E5%A4%A7%E6%A8%A1%E5%9E%8B%E5%9F%BA%E7%A1%80%E7%9F%A5%E8%AF%86/00-LLaMA) ｜ [LLaMA_1_paper](https://arxiv.org/abs/2302.13971) | [LLaMA_2_paper](https://arxiv.org/abs/2307.09288) | [LLaMA_3_paper](https://ai.meta.com/research/publications/the-llama-3-herd-of-models/)
- [x] [Qwen技术细节整理](https://github.com/sysu19351118/algorithm_code_summary/tree/master/LLM/%E5%A4%A7%E6%A8%A1%E5%9E%8B%E5%9F%BA%E7%A1%80%E7%9F%A5%E8%AF%86/01-Qwen)
- [ ] DeepSeek


## 搜广推
### 搜广推基础知识
- [x] [召回](https://www.notion.so/2165e95ca88980cb96f6ccc0d24727ae?source=copy_link)
- [x] [排序](https://www.notion.so/2175e95ca8898083be9ad966899298cb?source=copy_link)
### 经典实现
- [x] [DCN 深度交叉神经网络](https://github.com/sysu19351118/algorithm_code_summary/tree/master/RecommendationSystem/DCN)
- [x] [PPNet](https://github.com/sysu19351118/algorithm_code_summary/tree/master/RecommendationSystem/PPNet)
- [ ] SENet Bilinear
### 生成式召回热点
#### TYPE1 ZeroShot 形式引入LLM
- [x] LLMRank [paper](https://arxiv.org/pdf/2305.08845) | [code](https://github.com/RUCAIBox/LLMRank) | 利用llm的zeroshot能力进行排序
- [x] LLMRec [paper](https://dl.acm.org/doi/abs/10.1145/3616855.3635853) | [code](https://github.com/HKUDS/LLMRec.git.) | 利用LLM结合graph建模完成生成式推荐

#### TYPE Transformer based model + semantic ids 完成生成式召回任务
- [x] RQ-VAE | 提出了一种残差量化自编码器，可以将物品映射为具有层次关系的语义path
- [x] Tiger Google ｜ 基于RQVAE进行semantic id编码，将semantic id下挂载的物品作为召回结果进行召回
- [x] OneRec Kuaishou | 基于RQKmeans进行semantic编码，然后使用强化学习技术优化transformer模型的训练
- [x] 

# 工具整理
- [x] [常用库函数整理](https://github.com/sysu19351118/algorithm_code_summary/tree/master/ToolsLearning/01-%E5%B8%B8%E7%94%A8%E5%BA%93%E5%87%BD%E6%95%B0%E5%9F%BA%E7%A1%80%E7%9F%A5%E8%AF%86)
- [x] [pytorch lightning 训练框架](https://github.com/sysu19351118/algorithm_code_summary/tree/master/ToolsLearning/02-pytorch_lighting%E8%AE%AD%E7%BB%83%E6%A1%86%E6%9E%B6)
- [x] [pyspark大数据框架](https://github.com/sysu19351118/algorithm_code_summary/tree/master/RecommendationSystem/PySpark)
- [ ] lamma factory 大模型微调框架
- [ ] git 代码管理


