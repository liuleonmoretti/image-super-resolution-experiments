# 历史实验记录

## 数据来自哪里

本整理版按原路径保存三组 `params.json`、训练日志和验证指标 JSON，复制时未改写数值。下表的 best / last 指原工程保存的检查点类型；实际权重留在本地档案中，没有包含在此包。

| 模型 | best PSNR | best SSIM | last PSNR | last SSIM | 配置轮数 | batch size | 学习率 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| SRCNN | 27.319193 | 0.898265 | 27.319193 | 0.898265 | 300 | 512 | 0.00001 |
| DRRN | 28.982096 | 0.918785 | 28.976343 | 0.918868 | 100 | 64 | 0.0001 |
| GAN | 26.369649 | 0.871812 | 26.354661 | 0.872262 | 50 | 128 | 0.0001 |

CSV 保留 JSON 中的完整精度：[experiment-results.csv](../experiment-results.csv)。日志时间戳未作时区换算。

| 模型 | 参数 | 训练日志 | best 指标 | last 指标 |
| --- | --- | --- | --- | --- |
| SRCNN | [参数](../Super_Resolution/experiments/srcnn_model/params.json) | [日志](../Super_Resolution/experiments/srcnn_model/train.log) | [best](../Super_Resolution/experiments/srcnn_model/metrics_val_best_weights.json) | [last](../Super_Resolution/experiments/srcnn_model/metrics_val_last_weights.json) |
| DRRN | [参数](../Super_Resolution/experiments/drrn_model/params.json) | [日志](../Super_Resolution/experiments/drrn_model/train.log) | [best](../Super_Resolution/experiments/drrn_model/metrics_val_best_weights.json) | [last](../Super_Resolution/experiments/drrn_model/metrics_val_last_weights.json) |
| GAN | [参数](../Super_Resolution/experiments/gan_model/params.json) | [日志](../Super_Resolution/experiments/gan_model/train.log) | [best](../Super_Resolution/experiments/gan_model/metrics_val_best_weights.json) | [last](../Super_Resolution/experiments/gan_model/metrics_val_last_weights.json) |

## 数据准备的代码设置

`build_dataset.py` 默认对 218×178 图像进行中心裁剪，得到 144×144 清晰图像。随后按 4 倍缩小，得到 36×36 图像，再按 `input_size` 调整输入尺寸。数据划分使用种子 230，比例约为 98% / 1% / 1%。

- CNN：通常输入和目标都为 144×144，输入已经降采样并插值放大。
- GAN：生成器在训练入口设为 4 倍上采样，需要 36×36 输入对应 144×144 目标。不能直接沿用 CNN 的同尺寸数据而不核对。

完整原始数据、实际输入文件清单与最终环境依赖版本未随档案保留。代码中的默认设置不等于已确认的全部训练条件。

## 如何理解这些结果

日志末尾分别记录到 SRCNN 300 轮、DRRN 100 轮、GAN 50 轮。原工程按验证 PSNR 保存 best，因此 DRRN 的 last SSIM 略高于 best SSIM 并不矛盾。

这些记录可用于回顾当时实验，但尚未统一数据划分和评价实现重新计算。SSIM 的通道轴处理存在待复核项，不能把当前表格直接作为标准基准测试排名，也不能凭这些数字断定某类模型普遍更好。

原档案中 DRRN 与 GAN 两张 `final loss.jpg` 的哈希完全相同，因此未把它们作为两次独立实验的曲线放入整理版。原文件仍保留在本地，训练日志没有改写。
