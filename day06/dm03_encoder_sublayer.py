from dm02_encoder_element import *
from dm01_input import *

# todo 1. 初始化子层连接结构, 核心是: 残差连接 + 规范化层, 让模型训练更稳定, 避免梯度消失或者梯度爆炸.
class SublayerConnection(nn.Module):
    # 1. 初始化函数
    def __init__(self, d_model, dropout=0.1):
        """
        函数功能： 初始化子层链接结构
        :param d_model: 输入维度
        :param dropout: 随机失活概率
        """
        # 1. 初始化父类成员
        super().__init__()
        # 2. 定义规范化层
        self.norm = nn.LayerNorm(d_model)
        # 3. 定义随机失活层
        self.dropout = nn.Dropout(p=dropout)

    # 2. 前向传播函数
    def forward(self, x, sublayer):
        """
        函数功能： 完成具体的子层的计算动作
        :param x: 输入张量，形状[batch_size, seq_len, d_model]->[2,4,512]
        :param sublayer: 输入的子层对象，例如多头注意力对象，前馈全连接对象
        :return:
        """
        # 实现方式： 子层处理，层规范化，残差链接
        my_result = self.dropout(self.norm(sublayer(x)))+x
        # 返回结果
        return my_result


# todo 2. 测试子层连接结构.
def use_sublayer():
    # 1. 准备输入数据
    x = use_position()
    # 2. 实例化子层链接结构
    sublayer_conn = SublayerConnection(d_model=512)
    # 多头注意力子层
    # def sublayer(x):
    #     # 3.1 创建 多头注意力层对象(Multi-Head Attention)
    #     # multi_attn = MultiHeadAttention(d_model=512, h=8)
    #     # 3.2 计算 注意力, 并返回, 因为是自注意力, 所以: Q=K=V=x
    #     # return multi_attn(x, x, x)
    # result = sublayer_conn(x,lambda x:MultiHeadAttention(512,8)(x,x,x))
    # 前馈全连接子层
    result = sublayer_conn(x,lambda x:FeedForward(512,2048)(x))
    # 打印子层处理结果
    print(f"子层处理结果：{result.shape}")
