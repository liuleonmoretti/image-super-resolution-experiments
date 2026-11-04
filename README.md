# 图像超分辨率训练实验：SRCNN、DRRN 与 GAN

这是我基于已有模型和公开实现开展的图像超分辨率训练实验，整理了 SRCNN、DRRN 和 GAN 三种方法的源码、训练参数、日志与历史验证结果。

**实验与记录整理：[liuleonmoretti](https://github.com/liuleonmoretti)。** 我的主要工作是独立进行模型训练、实验执行和结果整理。模型架构及复用的实现归原作者；本项目不将这些基础模型描述为本人提出的新算法。

主要工程来源为 Yiyang Li、Yilun Xu、Ji Yu 的 [Super Resolution with CNNs and GANs](https://github.com/alohaleonardo/Super_Resolution_with_CNNs_and_GANs)。模型论文、代码来源和本地版本的差异分别列在 [参考资料](REFERENCES.md)、[第三方来源说明](THIRD_PARTY_NOTICES.md) 和 [来源比对记录](docs/PROVENANCE.md) 中。

## 实验内容

| 方法 | 本地实现的结构 | 阅读入口 |
| --- | --- | --- |
| SRCNN | 三层卷积，卷积核为 9×9、1×1、5×5；RGB 输入与输出 | [srcnn.py](Super_Resolution/model/srcnn.py) |
| DRRN | 128 通道、两个共享卷积、5 次递归，包含局部与全局残差连接 | [drrn.py](Super_Resolution/model/drrn.py) |
| GAN | 生成器含 5 个残差块和 PixelShuffle，训练入口使用 4 倍上采样，配合独立判别器 | [gan.py](Super_Resolution/model/gan.py) |

GAN 属于 SRGAN 方向的实现；这里描述实际保存的代码结构，不声称完整复现原论文的所有设置。

## 保存的验证结果

下表转录自各实验目录的 `metrics_val_best_weights.json` 与 `params.json`，不是整理时重新训练或复算得到的结果。`best` 按原工程的 PSNR 选取，不表示每个指标都分别达到最高。

| 模型 | PSNR | SSIM | 训练轮数 | Batch size | 学习率 |
| --- | ---: | ---: | ---: | ---: | ---: |
| SRCNN | 27.319193 | 0.898265 | 300 | 512 | 0.00001 |
| DRRN | 28.982096 | 0.918785 | 100 | 64 | 0.0001 |
| GAN | 26.369649 | 0.871812 | 50 | 128 | 0.0001 |

原日志记录于 2025 年 4 月 29—30 日。完整数值见 [CSV 汇总](experiment-results.csv) 与 [实验记录说明](docs/EXPERIMENTS.md)。数据划分和指标实现仍需复核，这些数值不用于宣称标准测试集排名或优于原论文。

## 目录与源码入口

```text
Super_Resolution/
  model/                   SRCNN、DRRN、GAN、数据加载及通用模块
  experiments/             三种模型的参数、训练日志和保存的指标
  pytorch_ssim/            随原工程保存的 SSIM 实现
  build_dataset.py         数据预处理与划分
  train_cnn.py             SRCNN / DRRN 训练入口
  train_gan.py             GAN 训练入口
  evaluate_cnn.py          CNN 评估
  evaluate_gan.py          GAN 评估
  img_cnn_eval.py          CNN 单图推理
  img_gnn_eval.py          GAN 单图推理，沿用原文件名
  loss.py                  GAN 损失函数
  logger.py                日志工具
  utils.py                 配置、检查点、指标工具
tools/PSNR/                 单独保存的图像评估工具
docs/                      实验说明、来源比对、复现注意事项
third_party_licenses/      查到的组件许可原文，适用范围见来源说明
```

建议从三个模型文件开始，随后阅读 [CNN 训练](Super_Resolution/train_cnn.py)、[GAN 训练](Super_Resolution/train_gan.py) 以及对应实验目录中的参数与日志。

## 运行状态

这是历史工程的源码与实验记录整理版，18 个 Python 文件保持原样。本次完成了语法解析和文件一致性检查，尚未恢复训练环境、加载权重或验证端到端推理。依赖版本、数据配对、SSIM 计算和旧日志接口的问题见 [复现说明](docs/REPRODUCIBILITY.md)。

原始数据集、示例图片与模型权重不包含在本整理版中。CelebA 应从 [官方项目页](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) 按其使用条件获取；本地保存的 8 个检查点仍在原档案中。

## 来源与许可

公开可见源码与拥有完整开源再分发许可是不同事项。主工程目前未找到明确的整体许可证；其 [许可证询问](https://github.com/alohaleonardo/Super_Resolution_with_CNNs_and_GANs/issues/5) 在核对时尚无回复。部分组件有 MIT 等许可，但不自动覆盖整个工程。

本项目按个人独立仓库组织，仓库由 liuleonmoretti 维护。采用独立仓库不改变第三方代码的原有归属和许可要求；本地修改版的完整再发布许可仍需进一步确认。本整理版未给第三方代码重新指定统一的 MIT / Apache 许可证，具体记录见 [LICENSE_STATUS.md](LICENSE_STATUS.md)。
