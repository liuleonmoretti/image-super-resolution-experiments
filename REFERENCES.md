# 参考论文、代码与数据

本项目的训练与实验由 liuleonmoretti 独立开展。以下作者分别提供基础模型、既有工程实现和数据资源；列为参考来源不表示他们直接参与了本次个人实验。

## 主要工程来源

Yiyang Li, Yilun Xu, Ji Yu. **Super Resolution with CNNs and GANs**, 2018.

- [项目仓库](https://github.com/alohaleonardo/Super_Resolution_with_CNNs_and_GANs)
- [用于比对的固定版本](https://github.com/alohaleonardo/Super_Resolution_with_CNNs_and_GANs/tree/e9d7dc4492f13bb404f1029b54c8db46e8764be2)

该仓库 README 列出了上述三位作者。其旧链接 `yiyang7/cs231n_proj` 当前指向同一仓库。源码比对表明，本地工程与该实现同源；无法仅凭源码确定当时下载的是此仓库还是其中间派生版本。

## 模型论文

1. **SRCNN**：Chao Dong, Chen Change Loy, Kaiming He, Xiaoou Tang. *Image Super-Resolution Using Deep Convolutional Networks.* [论文](https://arxiv.org/abs/1501.00092)
2. **DRRN**：Ying Tai, Jian Yang, Xiaoming Liu. *Image Super-Resolution via Deep Recursive Residual Network.* CVPR 2017. [论文与作者信息](https://openaccess.thecvf.com/content_cvpr_2017/html/Tai_Image_Super-Resolution_via_CVPR_2017_paper.html)
3. **SRGAN**：Christian Ledig, Lucas Theis, Ferenc Huszar, Jose Caballero, Andrew Cunningham, Alejandro Acosta, Andrew Aitken, Alykhan Tejani, Johannes Totz, Zehan Wang, Wenzhe Shi. *Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network.* CVPR 2017. [论文](https://arxiv.org/abs/1609.04802)

引用模型论文用于说明算法来源，不代表具体 Python 代码由论文作者提供，也不代表本地实现逐项采用论文中的设置。

## 代码组件

| 来源 | 对应关系 |
| --- | --- |
| [leftthomas/SRGAN](https://github.com/leftthomas/SRGAN) | GAN 残差块、上采样块以及部分生成器损失，与本地存在一致定义 |
| [CS230 Code Examples](https://github.com/cs230-stanford/cs230-code-examples) | 训练工程模板、配置和检查点工具；主工程 README 也明确引用 |
| [Po-Hsun-Su/pytorch-ssim](https://github.com/Po-Hsun-Su/pytorch-ssim) | Po-Hsun (Evan) Su 的 SSIM 实现，与本地多个函数一致 |
| [cszn/KAIR](https://github.com/cszn/KAIR) | Kai Zhang 的图像工具；本地独立 PSNR 工具中有大量一致函数 |
| [alterzero/DBPN-Pytorch](https://github.com/alterzero/DBPN-Pytorch) | 通用网络模块中的相同代码块参考；直接复制链路尚未确认 |
| [Michael Gygli 的 TensorBoard 日志示例](https://gist.github.com/gyglim/1f8dfb1b5c82627ae3efcfbbadb9f514) | 本地 `logger.py` 文件头明确标注的来源 |
| [PyTorch 数据加载教程](https://docs.pytorch.org/tutorials/beginner/data_loading_tutorial.html) | 本地数据加载器注释标注的教程来源之一 |

逐项证据和许可范围见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。同名函数或相同代码块只用于追溯来源，不据此认定某个候选仓库对整份文件拥有独占权利。

## 数据集

Ziwei Liu, Ping Luo, Xiaogang Wang, Xiaoou Tang. **Deep Learning Face Attributes in the Wild.** ICCV 2015.

[CelebA 官方页面、引用格式与使用条件](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)。工程默认预处理路径指向 CelebA 风格的人脸数据；实际训练用的完整文件清单未随档案保留。本整理版不重新分发该数据集。
