# NLP Practice

自然语言处理学习与代码练习，从 **2026 年 9 月 8 日**开始（本人确认），逐步学习文本预处理、词向量、循环神经网络、注意力机制和 Transformer。

前期基础内容推进较快；注意力机制和 Transformer 属于重难点，学习节奏较慢，投入更多时间理解公式与代码。日期回顾见 [学习记录](LEARNING_LOG.md)。目录中的 `dayXX` 是学习单元编号，不对应实际日历日期；目前没有 `day05` 练习目录。

## 练习内容

| 目录 | 内容 |
| --- | --- |
| `day01` | jieba 分词、自定义词典、词性标注、One-hot 编码 |
| `day02` | fastText CBOW 词向量、Embedding 与 TensorBoard、文本统计与词云、N-gram、chain/zip |
| `day03` | 序列补齐与截断、RNN、LSTM、GRU 的调用及张量形状 |
| `day04` | 批量矩阵乘法、基于 Q/K/V 的注意力机制 |
| `day06` | Transformer 输入层、掩码、缩放点积与多头注意力、前馈网络、规范化与残差连接、编码器层与编码器堆叠、单个解码器层 |

## 运行方式

这些文件是独立的学习脚本。涉及 `./data`、`./model` 或 `./runs` 的脚本需要在对应的 `dayXX` 目录中运行，例如：

```powershell
cd day01
python dm01_jieba1分词.py
```

根据脚本使用的模块安装依赖：`jieba`、`torch`、`tensorflow`、`tensorboard`、`fasttext`、`numpy`、`pandas`、`matplotlib`、`seaborn`、`wordcloud`。当前没有锁定依赖版本，也未验证全新环境下的完整运行。

## 数据与本地资源

- 仓库保留自定义词典，以及文本分析使用的 `day02/data/train.tsv` 和 `dev.tsv`。
- fastText 练习需要自行准备本地语料 `day02/data/wh02ad`；训练前创建 `day02/model` 目录，生成的模型为 `wh02_file.bin`。
- 词云练习需要在 `day02/data/SimHei.ttf` 放置可使用的中文字体，或自行修改脚本的字体路径。
- 下载的课程资料、模型、大型语料、字体、TensorBoard 日志和 IDE 设置不纳入 Git；原始本地文件保留。
- 本项目为个人学习练习，参考课程编写；随附数据来自本地课程练习材料。

## 当前练习状态

截至 **2026-10-25**，已编写三层编码器示例和单个解码器层，尚未完成整个解码器堆叠、输出生成层、完整 Transformer 及训练流程。

本次在本机 `nlpbase` 环境检查：21 个 Python 文件通过语法解析；编码器示例输出 `[2, 4, 512]`；单个解码器层在目标长度为 3、源长度为 4 时输出 `[2, 3, 512]`，数值均为有限值。前向运行通过不代表模型逻辑和训练正确性已验证。

保留学习代码当前状态，以下内容仍待完善：

- `padding_custom` 是占位函数；早期 `bmm` 示例的两个输入批次大小不同。
- `attention()` 使用 `-1e-9` 填充被屏蔽位置，该值接近零，不能有效屏蔽注意力；全零掩码示例也需要重新检查语义。
- 自定义 `LayerNorm` 的缩放和平移使用普通张量，尚未注册为可训练参数；标准差计算与标准 LayerNorm 实现也需对照。
- `use_sublayer()` 调用未定义的 `use_position()`；残差与规范化顺序仍需核对所参考的 Transformer 结构。
- `DecoderLayer` 克隆了四个子层连接，却只使用三个；尚未验证因果掩码与训练行为。

