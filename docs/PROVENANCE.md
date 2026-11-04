# 源码来源与本地版本差异

核对日期：2026-11-03。

## 比对方法

将本地工程的 14 个对应 Python 文件与上游 [固定版本 e9d7dc4](https://github.com/alohaleonardo/Super_Resolution_with_CNNs_and_GANs/tree/e9d7dc4492f13bb404f1029b54c8db46e8764be2) 比较。文本比较统一换行并忽略末尾空行；可解析的代码还比较 Python 抽象语法树（AST），忽略行号和注释。

上游训练和评估脚本使用旧式 `async=True` 参数，不能由本次的 Python 3.14 解析；这些文件使用文本差异核对，未假称完成 AST 相同判断。源码同源证据不能确定当时的下载地址、中间派生仓库或每一处改动的作者。

## 明确一致的部分

| 文件 | 比对结果 |
| --- | --- |
| `model/srcnn.py` | `Net`、`loss_fn`、`accuracy` 定义与上游 AST 相同 |
| `model/drrn.py` | `Net`、`loss_fn`、`accuracy` 定义与上游 AST 相同，5 次递归也存在于该上游实现 |
| `model/gan.py` | `Generator`、`Discriminator`、`ResidualBlock`、`UpsampleBLock`、`psnr` 定义与上游 AST 相同 |
| `loss.py` | `GeneratorLoss`、`GeneratorLoss_adv`、`GeneratorLoss_ssim`、`GeneratorLoss_notv`、`TVLoss` 五个类均与上游 AST 相同 |
| `model/base_networks.py` | 整个文件的 AST 与上游相同 |
| `model/data_loader.py` | 统一换行与末尾空行后文本完全相同 |
| `pytorch_ssim/__init__.py` | 统一换行与末尾空行后文本完全相同 |

因此，基础模型结构、DRRN 的这份递归设置以及已有的 SSIM/TV 等损失变体，应归到引用的实现中，不能作为本次个人实验提出的新结构或新损失。

## 本地版本保留的差异

- 训练入口的模型映射缩减为 SRCNN、DRRN、GAN。
- CNN 训练中的旧式 CUDA 参数改为 `non_blocking=True`。
- GAN 训练区分判别器更新时的 `fake_imgs.detach()` 与生成器更新时重新计算的判别器输出，并调整了训练指标统计。
- 部分评价函数更新了 `skimage.metrics` 导入与参数。
- 数据预处理中增加保存前的 `img_as_ubyte` 转换。
- 增加或保留单图推理脚本及独立 PSNR 工具。所比对主工程版本中没有找到两个单图入口；未进一步确定它们的直接来源。
- 保留三组本地训练参数、2025 年训练日志及指标。三组 best 指标均不同于所查上游版本中对应目录的数值；这一差异本身不用于推断完整训练过程。

本项目以训练和实验工作为主要个人贡献；上述差异用于说明保存下来的工程版本，不把全部差异自动等同于独创算法。

## 整理边界

本次复制保留 18 个 Python 文件及 12 个参数、指标和日志文件，均与本地来源逐字节一致。未修改模型、训练循环、实验日志或数值。第三方许可与新增说明单独保存。

未收录的内容：8 个模型检查点、原始与示例图片、重复曲线图、包含个人电脑路径的旧整理报告。所有原始文件仍留在原目录。
