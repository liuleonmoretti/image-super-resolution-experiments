# 第三方来源与署名

核对日期：2026-11-03。此文档记录来源与已观察到的许可证，不代替许可证本身，也不将尚未确认许可的代码重新授权。

## 主工程

- 名称：**Super Resolution with CNNs and GANs**。
- 作者：**Yiyang Li、Yilun Xu、Ji Yu**，依上游 README 署名。
- 仓库：[alohaleonardo/Super_Resolution_with_CNNs_and_GANs](https://github.com/alohaleonardo/Super_Resolution_with_CNNs_and_GANs)。
- 比对版本：`e9d7dc4492f13bb404f1029b54c8db46e8764be2`。
- 本地对应：`Super_Resolution/model/`、训练与评估脚本、预处理、损失函数、日志和配置工具等。
- 整体许可：未在该版本的完整文件树、README 或许可证问题回复中找到明确的统一开源许可。[上游 License 问题 #5](https://github.com/alohaleonardo/Super_Resolution_with_CNNs_and_GANs/issues/5) 核对时为 open、0 条回复。

本项目将其作为基础工程署名。基础模型及已有损失变体不计为本人提出的新算法；本人工作的定位是基于既有模型开展训练、实验与结果记录。

## 可追溯组件

| 本地内容 | 署名与上游 | 已核实情况 | 许可记录 |
| --- | --- | --- | --- |
| `model/gan.py`、`loss.py` 的部分定义 | Hao Ren / [leftthomas/SRGAN](https://github.com/leftthomas/SRGAN) | `ResidualBlock`、`UpsampleBLock`、`GeneratorLoss`、`TVLoss` 与对应上游定义的 AST 相同 | 上游 MIT，原文保存在 [SRGAN_MIT.txt](third_party_licenses/SRGAN_MIT.txt)；不据此覆盖主工程增加的全部内容 |
| 配置与训练工具 | [CS230 Teaching team](https://github.com/cs230-stanford/cs230-code-examples) | `Params`、`set_logger`、`save_dict_to_json` 等定义相同；主工程明确引用该模板 | [CS230_MIT.txt](third_party_licenses/CS230_MIT.txt) 保留团队、助教和教师署名 |
| `pytorch_ssim/__init__.py` | Po-Hsun (Evan) Su / [pytorch-ssim](https://github.com/Po-Hsun-Su/pytorch-ssim) | `SSIM`、`_ssim`、`create_window`、`gaussian` 四个定义相同 | 上游 `LICENSE.txt` 仅写有 `MIT`，原样记录于 [pytorch_ssim_LICENSE.txt](third_party_licenses/pytorch_ssim_LICENSE.txt)，未自行补造版权年份 |
| `tools/PSNR/utils/utils_image.py` | Kai Zhang / [KAIR](https://github.com/cszn/KAIR) | 44 个本地顶层函数中，31 个与所查版本 AST 相同，包括 PSNR、SSIM、插值和图像转换函数 | [KAIR_MIT.txt](third_party_licenses/KAIR_MIT.txt)；尚未确定本地是经 KAIR 还是另一个同源项目取得 |
| `model/base_networks.py` 的公共模块 | Muhammad Haris / [DBPN-Pytorch](https://github.com/alterzero/DBPN-Pytorch) | `DenseBlock`、`Upsample2xBlock` 与所查版本一致；其余模块存在差异 | [DBPN_MIT_reference.txt](third_party_licenses/DBPN_MIT_reference.txt) 仅作为该来源的许可记录，不能单凭两段一致实现给整份文件指定 MIT |
| `logger.py` | Michael Gygli / [TensorBoard 日志 Gist](https://gist.github.com/gyglim/1f8dfb1b5c82627ae3efcfbbadb9f514) | 本地文件头明确引用；本地为改动后的版本 | Gist 文件头写有 `BSD License 2.0`，未附完整条款；不擅自改写成某个确定的 BSD SPDX 标识 |
| `model/data_loader.py` | PyTorch 教程及主工程作者 | 本地注释明确引用数据加载与 Neural Style 教程；整个文件与主工程版本一致 | 保留原注释；教程引用不自动授予主工程自有部分的许可 |

## 如何标注贡献

- 实验执行与个人结果整理：liuleonmoretti。
- 基础工程、模型和组件：按上文及 [参考资料](REFERENCES.md) 列出原作者。
- GitHub 提交记录应对应实际提交；参考论文的作者不需要通过虚构提交或 `Co-authored-by` 变成个人实验的共同提交者。
- 这些引用不表示原作者审核、参与或认可了本次实验。

本地复制的源码未移除已有注释，新增说明集中放在此文件。主工程整体许可仍待确认，参见 [许可状态](LICENSE_STATUS.md)。
