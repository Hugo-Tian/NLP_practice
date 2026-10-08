import torch
import torch.nn as nn


# todo 1.GRU模型代码
def dm_gru_model():

    # 1. 实例化RNN模型
    # 参1： 词向量维度（输入维度）， 参2： 隐藏层维度(输出维度）  参3：隐藏层层数
    gru = nn.GRU(5,6,1)
    # 参1： 句子长度，  参2：批次大小，  参3： 词向量维度
    input = torch.randn(10,3,5)
    # 参1： 隐藏层层数  参2：批次大小  参3：隐藏层维度
    h0 = torch.randn(1,3,6)

    # 运行RNN模型
    # 本次的输出， 本次的隐藏状态 = rnn(本次的输入， 上一时刻隐藏状态）
    output, hn = gru(input,h0)

    print(f'输出形状：{output.shape}')
    print(f'隐藏层形状：{h0.shape}')
    print(f'输入形状：{input.shape}')
    print(gru)


if __name__ == '__main__':
    dm_gru_model()