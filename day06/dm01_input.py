import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import math


# todo 1 定义类，模拟词嵌入层
class Embedding(nn.Module):
    # 1.初始化函数
    def __init__(self,vocab_size,d_model):
        '''

        :param vocab_size: 词汇表大小
        :param d_model:  词嵌入的维度
        '''
        # 初始化父类信息
        super().__init__()
        # 定义变量
        self.vocab_size = vocab_size
        self.d_model = d_model
        # 定义词嵌入层
        self.embed = nn.Embedding(vocab_size,d_model)


    # 2. 前向传播
    def forward(self,x):
        # 将输入的单词映射为词向量，乘根号d_model进行缩放
        # 按 Transformer 的做法缩放词嵌入，调整其与位置编码相加时的相对数值尺度
        return self.embed(x)* math.sqrt(self.d_model)


# todo 2 测试词嵌入层代码
def dm01_embedding():
    # 词表大小
    vocab_size = 1000
    # 词嵌入维度
    d_model = 512
    # 创建词嵌入对象
    my_embed = Embedding(vocab_size,d_model)
    # 创建张量
    x=torch.tensor([
        [200,300,203,403],
        [143,765,343,754]
    ])
    # 前向传播，获得词嵌入向量
    result = my_embed(x)
    print(result)
    print(result.shape)

# todo 3 定义类，模拟位置编码层
class PositionalEncoding(nn.Module):
    # 1. 初始化函数
    def __init__(self,d_model,dropout=0.1,max_len=60):
        """
        :param d_model: 词向量维度
        :param dropout: 随机失活概率
        :param max_len: 最大句子长度
        """
        # 初始化父类信息
        super().__init__()
        # 定义dropout层，防止过拟合
        self.dropout = nn.Dropout(p=dropout)
        # 定义pe， 用于保存位置编码结果
        pe = torch.zeros(max_len,d_model)
        # 定义一个列向量，范围：0 ~ max_len - 1
        position = torch.arange(0,max_len).unsqueeze(1)

        # 定义一个变化矩阵，本质是：公式里的1/10000^(2i/d_model)
        # 1/(10000^(2i/d_model))= e^((2i/d_model)*-ln(10000))
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        print(f'div_term:{div_term.shape}')
        # position形状：[max_len,1],div_term形状：[1,256]
        position_value = position*div_term
        # 进行pe赋值，偶数位置使用正弦函数
        pe[:,0::2] = torch.sin(position_value)
        # 进行pe赋值，奇数位置使用余弦函数
        pe[:,1::2] = torch.cos(position_value)

        # 升维，形状【1，60，512】
        pe = pe.unsqueeze(0)
        # 把pe注册到模型缓冲区
        self.register_buffer('pe',pe)

    # 2. 前向传播
    def forward(self,x):
        # x : 词向量，形状：[batch_size,seq_len,d_model]->[1,60,512]
        # 把词向量与位置编码相加
        x = x+self.pe[:,:x.size(1)]
        return self.dropout(x)


# todo 4. 测试PositionalEmbedding(位置编码层）
def dm02_position():
    # 定义词汇表大小和词嵌入维度
    vocab_size = 1000
    d_model = 512
    # 实例化Embedding层
    my_embed = Embedding(vocab_size,d_model)
    # 创建输入张量
    x=torch.tensor([
        [200,300,203,403],
        [143,765,343,754]

    ])
    # 计算词向量结果
    embed_x = my_embed(x)
    # 创建位置编码层对象
    my_position = PositionalEncoding(d_model,dropout=0.1)
    # 计算位置编码结果
    position_x = my_position(embed_x)
    print(position_x.shape)

if __name__ == '__main__':
    # dm01_embedding()
    dm02_position()