import math
import numpy as np
import copy
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from dm01_input import *



# todo 1 测试上三角矩阵
def dm01_test_triu():
    arr=[
        [1,1,1,1,1],
        [2,2,2,2,2],
        [3,3,3,3,3],
        [4,4,4,4,4],
        [5,5,5,5,5],
    ]

    # 分别测试：k=0,k=-1,k=1的情况
    print(np.triu(arr,0))
    print(np.triu(arr,1))
    print(np.triu(arr,-1))

# todo 2 构建下三角矩阵
def dm02_test_triu2(size):
    temp=np.triu(np.ones((1,size,size)),1).astype('uint8')

    return torch.from_numpy(1-temp)


# todo 4 定义函数，进行注意力的计算
def attention(query, key, value, mask=None, dropout=None):
    """
    :param query:查询向量
    :param key: 键向量
    :param value: 值向量
    :param mask: 是否掩码处理
    :param dropout: 是否随机失活
    :return: 返回的是注意力输出（融合的信息）和注意力权重（用于可视化，调试）
    """
    # 1.定义查询张量Q的特征维度
    d_k= query.size(-1)
    # 2.计算原始注意力分数，即 Q * K^t/√d_k
    scores = torch.matmul(query, key.transpose(-1,-2)) / math.sqrt(d_k)
    # 3.掩码处理（可选）
    if mask is not None:
        scores= scores.masked_fill(mask==0, -1e-9)

    # 4. 计算注意力权重。
    # dim=-1,在最后1个维度上进行归一化，让每个query位置权重总和为1
    p_attn= F.softmax(scores, dim=-1)

    # 5. 随机失活（可选）
    if dropout is not None:
        p_attn= dropout(p_attn)

    # 6. 计算最终的注意力输出：权重加权求和。
    return torch.matmul(p_attn, value), p_attn

# todo 5 测试注意力机制
def use_attention():
    # 获取位置编码处理后的结果（词嵌入层结果 + 位置编码结果）
    position_x = dm02_position()
    # Q,K,V使用同一个张量
    Q=K=V=position_x
    # 没有掩码，直接调用attention
    result1 , p_attn = attention(Q,K,V)
    print(f'result1: {result1}')
    print(f'p_attn: {p_attn}')

    # 构造一个全0的掩码张量
    mask= torch.zeros(2,4,4)
    # 带掩码，调用attention()
    result2,p_attn = attention(Q,K,V,mask=mask)
    print(f'result2: {result2}')
    print(f'p_attn: {p_attn}')

# todo 6 定义一个克隆模块 ->实现克隆指定的模块N次，返回N个相同模块的列表
def clones(module, N):
    """
    函数功能： 创建N个相同的模块，返回一个列表，深拷贝，每个模块拥有独立的参数
    :param module: 要被克隆的模块
    :param N: 克隆的次数
    :return:  具有N个相同模块的列表
    """
    # return nn.Modulelist([具体深拷贝动作 for _ in range(N)])
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

# # todo 7 定义函数，实现多头注意力机制
# class MultiHeadAttention(nn.Module):
#     # 参数初始化
#
#     def __init__(self,embed_dim,head,dropout_p=0.1):
#         """
#
#         :param embed_dim:  词向量维度
#         :param head:  头数
#         :param dropout_p:
#         """
#         # 初始化父类成员
#         super.__init__()
#
#         # 确保词向量维度数量能被头数整除
#         assert ...
#         # 计算分头后的词嵌入维度
#         self.d_k=...
#         self.head = ...
#         # 定义4个线性层，前3个负责：Q/K/V的投影，最后1个负责：输出多头注意力机制的结果
#         self.linear = ...
#         # 定义随机失活层
#         self.dropout = ...
#         # 保存注意力权重， 用于可视化或者分析。
#         self.attn = ...
#
#     # 前向传播，实现多头注意力计算流程
#     def forward(self,query,key,value,mask=None):
#         # 1.判断是否掩码
#         # Q/K/V 的形状：[batch_size,seq_len,d_model],例如【2，4，512】
#         # mask的形状：[batch_size,seq_len,seq_len],例如【1，4，4，】
#         if...:
#             mask=...
#         # 获取batch_size的批量大小
#         self.batch=...
#
#         # 线性变换：Q,K,V -> 多头注意力计算
#         # model(x):     通过线性层 将输入映射到embed_dim的维度：512
#         # view(...):    把输入张量转成[batch_size,seq_len,head,d_k]-> [2,4,8,64] --> [ batch_size,head,seq_len,d_k] ->[2,8,4,64]
#         query,key,value = [
#             ...
#         ]
#
#         # 多头注意力计算
#         # x的形状（注意力输出）：   [batch_size,head,seq_len,d_k] -> [2,8,4,64]
#         # self.att(注意力权重):   [batch_size,head,seq_len,seq_len] -> [2,4,8,8]
#         x,self.attn = ...
#         # 将多头注意力结果合并
#         # 为什么要加contiguous：。。。
#         attn_x = ...
#
#         # 通过最后一个线性层处理，返回结果。
#         return...
#

# todo 7 定义函数，实现多头注意力机制
class MultiHeadAttention(nn.Module):
    # 参数初始化
    def __init__(self, embed_dim, head, dropout_p=0.1):
        """

        :param embed_dim:  词向量维度
        :param head:  头数
        :param dropout_p: 随机失活的概率
        """
        # 初始化父类成员
        super().__init__()

        # 确保词向量维度数量能被头数整除
        assert embed_dim%head==0
        # 计算分头后的词嵌入维度
        self.d_k = embed_dim//head
        self.head = head
        # 定义4个线性层，前3个负责：Q/K/V的投影，最后1个负责：输出多头注意力机制的结果
        self.linears = clones(nn.Linear(embed_dim, embed_dim), 4)
        # 定义随机失活层
        self.dropout = nn.Dropout(p=dropout_p)
        # 保存注意力权重， 用于可视化或者分析。
        self.attn = None

    # 前向传播，实现多头注意力计算流程
    def forward(self, query, key, value, mask=None):
        # 1.判断是否掩码
        # Q/K/V 的形状：[batch_size,seq_len,d_model],例如【2，4，512】
        # mask的形状：[batch_size,seq_len,seq_len],例如【1，4，4】
        if mask is not None:
            mask = mask.unsqueeze(1)
        # 获取batch_size的批量大小
        self.batch = query.size(0)

        # 线性变换：Q,K,V -> 多头注意力计算
        # model(x):     通过线性层 将输入映射到embed_dim的维度：512
        # view(...):    把输入张量转成[batch_size,seq_len,head,d_k]-> [2,4,8,64] --> [ batch_size,head,seq_len,d_k] ->[2,8,4,64]
        # -1 表示这一维大小由Pytorch自动推算
        #
        query, key, value = [
            model(x).view(self.batch, -1, self.head, self.d_k).transpose(1, 2)
            for model, x in zip(self.linears,(query, key, value))
        ]

        # 上述代码等价于：
        # results = []
        #
        # for model, x in zip(self.linears, (query, key, value)):
        #     x = model(x)  # 线性变换
        #     x = x.view(self.batch, -1, self.head, self.d_k)  # 拆成多个头
        #     x = x.transpose(1, 2)  # 把头维度放到前面
        #     results.append(x)
        #
        # query, key, value = results  # 将三个结果分别赋值

        # 多头注意力计算
        # x的形状（注意力输出）：   [batch_size,head,seq_len,d_k] -> [2,8,4,64]
        # self.att(注意力权重):   [batch_size,head,seq_len,seq_len] -> [2,4,8,8]
        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)
        # 将多头注意力结果合并
        # 为什么要加contiguous：让张量按照当前维度顺序，在内存中连续排列，方便后面使用view()改变形状
        attn_x = x.transpose(1,2).contiguous().view(self.batch, -1, self.head*self.d_k)

        # 通过最后一个线性层处理，返回结果。
        return self.linears[-1](attn_x)

# todo 8 测试多头注意力机制
def use_multihead():
    # 1. 创建多头注意力对象
    my_attention = MultiHeadAttention(512,8)
    # 2. 获取位置编码处理后的结果（词嵌入层结果 + 位置编码结果）
    position_x = dm02_position()
    # 3. 定义遍历，表示QKV
    query = key = value = position_x
    # 4. 创建掩码张量，形状是：[batch_size, seq_len, seq_len]
    mask = torch.zeros(query.size(0),query.size(1),key.size(1))
    # 5. 将数据推送给模型，获得多头注意力结果。
    result = my_attention(query, key, value, mask=mask)
    # 打印输出形状
    print(f'result: {result.shape}')
    # 返回结果
    return result

# todo 9 初始化前馈全连接层（包含两个全连接层）
# 目的：对注意力层的结果进一步加工（强化特征）

class FeedForward(nn.Module):
    # 初始化函数
    # 参1：输入数据的词向量维度   参2：前馈全连接层的维度   参3：随机失活概率
    def __init__(self, d_model, d_ff,dropout_p=0.1):
        # 1. 初始化父类成员
        super().__init__()
        # 2. 定义一个全连接层，输入维度：d_model,输出维度：d_ff
        self.linear1 = nn.Linear(d_model, d_ff)
        # 3. 定义一个全连接层，输入维度：d_ff,输出维度：d_model
        self.linear2 = nn.Linear(d_ff, d_model)
        # 4. 创建随机失活层
        self.dropout = nn.Dropout(p=dropout_p)


    # 前向传播
    def forward(self, x):
        # 1. 通过第1个全连接层，映射到d_ff的维度
        x=self.linear1(x)
        # 2. 通过激活函数处理，再通过随机失活处理
        x=self.dropout(F.relu(x))
        # 3. 通过第二个全连接层，映射到d_model的维度
        x=self.linear2(x)
        # 4. 返回
        return x


# todo 10. 测试前馈全连接层 -> 先拿到多头注意力层结果, 把结果送给前馈全连接层处理.
def use_ff():
    # 1. 获得多头注意力层的结果
    attn_x = use_multihead()
    # 2. 创建前馈全连接层对象
    my_ff = FeedForward(512, 2048, dropout_p=0.1)
    # 3. 将多头注意力层的结果，送给前馈全连接层处理。
    result = my_ff(attn_x)
    # 输出并返回结果
    print(result.shape)
    return result

# todo 11, 定义1个规范化层，初始化规范化层(Layer Normalization), 用于对模型张量进行标准化处理，让模型训练更稳定
class LayerNorm(nn.Module):
    # 1. 初始化函数
    def __init__(self, features, eps=1e-12):
        """
        函数作用：
        :param features:词嵌入维度：512
        :param eps: 小常数，防止分母变为0
        """
        # 初始化父类成员
        super().__init__()
        # 线性公式： y = kx+b ->ax+b
        # 定义缩放系数，作用：对标准化后的数据进行缩放
        # 初始化为1，形状为[features]
        self.a = torch.ones(features)
        # 定义偏置系数（平移系数），作用：对标准化后的数据进行平移
        # 初始化为0，形状为[features]
        self.b = torch.zeros(features)
        # 定义eps:小常数，防止分母变为零
        self.eps = eps

    # 2. 前向传播
    def forward(self, x):
        # x的形状：[batch_size,seq_len,d_model]->[2,4,512]
        # 计算词向量维度的均值
        x_mean= x.mean(-1, keepdim=True)

        # 计算最后一维的标准差
        x_std = x.std(-1, keepdim=True)

        # 标准化处理
        return self.a* (x-x_mean) / (x_std + self.eps) + self.b

# todo 12 测试规范化层，思路：拿到多头注意力机制结果 -> 把该结果传入前馈全连接进行处理 -> 把处理结果传入规范化层处理 -> 返回结果
def use_layer_norm():
    # 1. 获取前馈全连接层结果
    ff_x= use_ff()
    # 2. 创建规范化层对象。
    my_layer_norm = LayerNorm(512)
    # 3. 将前馈全连接层结果传给规范化层
    result = my_layer_norm(ff_x)
    # 4. 输出结果
    print(f'(规范化层)result: {result.shape}')



if __name__=='__main__':
    # use_attention()
    # print(dm02_test_triu2(5))
    # use_multihead()
    # use_ff()
    use_layer_norm()