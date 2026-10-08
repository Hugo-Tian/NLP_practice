import torch
import torch.nn as nn
import torch.nn.functional as F


#定义模块
class MyAttn(nn.Module):
    # 1.初始化函数
    def __init__(self,query_size,key_size,value_size1,value_size2,output_size):
        # 初始化父类内容
        super().__init__()
        # 查询张量的维度
        self.query_size=query_size
        # 键张量的维度
        self.key_size=key_size
        # 词张量序列长度
        self.value_size1=value_size1
        # 词张量维度
        self.value_size2=value_size2
        # 输出张量维度
        self.output_size=output_size

        # 注意力权重计算层，将Q和K拼接后，映射到：值序列长度维度
        self.attn = nn.Linear(self.query_size+self.key_size,self.value_size1)
        # 注意力融合层：将原始Q和注意力加权后的V拼接融合后，映射到：输出维度
        self.attn_combine = nn.Linear(self.query_size+self.value_size2,self.output_size)

    # 2.前向传播
    def forward(self,Q,K,V):
        # 2.1 计算注意力权重
        # 拼接q和k，维度：【1，1，32】+【1，1，32】=【1，1，64】
        qk_cat = torch.cat((Q[0],K[0]),dim=-1)
        # 通过线性层计算注意力得分，维度：【1，64】-》【1，32】
        attn_scores = self.attn(qk_cat)
        # softmax计算概率，【1，32】-》【1，32】
        attn_weights = F.softmax(attn_scores,dim=-1)

        # 2.2 应用注意力权重（注意力分配系数）到值张量V
        # 拓展注意力权重维度，以便匹配V的批次维度，即【1，32】-》【1，1，32】
        attn_weights_expanded = attn_weights.unsqueeze(0)
        # 使用bmm（）函数执行矩阵乘法，维度变化：【1，1，32】 * 【1，32，64】 = 【1，1，64】
        attn_applied = torch.bmm(attn_weights_expanded,V)

        # 2.3 拼接Q和V后输出结果
        # 拼接Q和V 【1，1，32】+ 【1，1，64】=【1，1，96】
        output_cat = torch.cat((Q,attn_applied),dim=-1)
        # 通过线性层输出维度， 【1，1，96】 - 》【1，1，32】
        output = self.attn_combine(output_cat)
        # 2.4 返回结果
        return output,attn_weights

if __name__=='__main__':
    query_size,key_size,value_size1,value_size2,output_size=32,32,32,64,32

    Q=torch.randn(1,1,query_size)
    K=torch.randn(1,1,key_size)
    V=torch.randn(1,value_size1,value_size2)

    my_attn=MyAttn(query_size,key_size,value_size1,value_size2,output_size)
    output,attn_weights=my_attn(Q,K,V)

    print('=·='*40)
    print(output)
    print(attn_weights)