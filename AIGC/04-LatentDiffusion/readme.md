## LDM 实现 Miniset重建

Latent Diffusion Model (LDM) 是一种高效的生成模型，结合了扩散模型（Diffusion Model）和隐空间（Latent Space）技术，用于生成高质量图像或其他数据（如文本、音频）。其核心思想是在低维隐空间中进行扩散过程，而非原始高维数据空间，从而显著降低计算成本，同时保持生成质量。他的核心结构就是VAE+Diffusion


运行方式：
```bash
python main.py
```
效果：

<img src="image.png" alt="alt text" width="100">


## 优秀的博客：
* [生成扩散模型漫谈（一）：DDPM = 拆楼 + 建楼](https://spaces.ac.cn/archives/9119)
* [生成扩散模型漫谈（二）：DDPM = 自回归式VAE](https://spaces.ac.cn/archives/9152)
* [生成扩散模型漫谈（三）：DDPM = 贝叶斯 + 去噪](https://spaces.ac.cn/archives/9164/comment-page-3#comments)


## 待整理：
DDPM公式手推

## 难点问题整理：
* 为什么前向传播之需要加噪一次，而推理时反向去噪要去噪T次？ 看上面的漫谈3

