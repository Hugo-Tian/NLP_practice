# NLP Practice

自然语言处理学习与代码练习，约从 **2026 年 9 月 3 日**开始，逐步学习文本预处理、词向量、循环神经网络、注意力机制和 Transformer。

日期回顾见 [学习记录](LEARNING_LOG.md)。目录中的 `dayXX` 是学习单元编号，不对应实际日历日期；目前没有 `day05` 练习目录。

## 练习内容

| 目录 | 内容 |
| --- | --- |
| `day01` | jieba 分词、自定义词典、词性标注、One-hot 编码 |
| `day02` | fastText CBOW 词向量、Embedding 与 TensorBoard、文本统计与词云、N-gram、chain/zip |
| `day03` | 序列补齐与截断、RNN、LSTM、GRU 的调用及张量形状 |
| `day04` | 批量矩阵乘法、基于 Q/K/V 的注意力机制 |
| `day06` | Transformer 输入层：缩放词嵌入与正弦/余弦位置编码 |

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

保留学习过程中的代码原貌，部分练习尚待完善：`padding_custom` 仍是占位函数；`bmm` 示例的两个输入批次大小不同，直接运行会报错；位置编码 `forward` 的切片目前使用词向量维度，后续应检查序列长度维度。Transformer 当前只包含输入层练习，并非完整模型。
