import torch
import torch.nn as nn

def lstm_model():
    # 参1：输入维度   参2：隐藏层维度（输出维度）    参3：LSTM层数    参4：是否双向
    lstm=nn.LSTM(input_size=5, hidden_size=6, num_layers=1)

    input=torch.randn(1,3,5)
    h0=torch.randn(1,3,6)
    c0=torch.randn(1,3,6)

    # 必须加括号
    output,(hn,cn)=lstm(input,(h0,c0))

    print(f'输出形状：{output.shape}')
    print(f'隐藏层形状：{h0.shape}')
    print(f'输入形状：{input.shape}')
    print(f'细胞状态形状：{c0.shape}')


if __name__=='__main__':
    lstm_model()