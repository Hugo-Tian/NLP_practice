# 导包
from dm01_input import *                # 输入部分
from dm02_encoder_element import *      # 组件
from dm03_encoder_sublayer import *     # 子层

# todo 1. 初始化编码器层 -> 所谓的编码器层, 就是把刚才的两个子层合二为一串起来.
class EncoderLayer(nn.Module):
    # 1. 初始化函数
    def __init__(self,d_model,self_attn,feed_forward, dropout=0.1):
        """
        函数功能：初始化函数
        :param d_model: 词嵌入维度：512
        :param self_attn: 多头注意力（对象）
        :param feed_forward: 前馈全连接（对象）
        :param dropout: 随机失活概率
        """
        # 1. 初始化父类成员
        super().__init__()
        # 2. 保存子层实例
        self.d_model = d_model
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        # 3. 克隆两个子层链接结构
        self.sublayer = clones(SublayerConnection(d_model,dropout),2)

    # 2. 前向传播函数
    def forward(self,x,mask):
        # 1. 第一层：子层链接：自注意力层
        x = self.sublayer[0](x,lambda x:self.self_attn(x,x,x,mask))
        # 2. 第2层 子层连接： 前馈全连接层
        x = self.sublayer[1](x,lambda x:self.feed_forward(x))
        # 返回结果
        return x

# todo 2. 测试编码器层
def use_encoder_layer():
    # 1. 准备数据 -> 位置编码层处理后的数据
    x = dm02_position()
    # 2. 实例化子层对象
    # 2.1 多头注意力对象
    multi_head = MultiHeadAttention(512,8)
    # 2.2 前馈全连接对象
    feed_forward = FeedForward(512,2048)
    # 3. 创建编码器层对象
    encoder_layer = EncoderLayer(512,multi_head,feed_forward)
    # 4. 构建掩码张量， 形状：[batch_size, seq_len, seq_len]
    mask = torch.zeros(2,4,4)
    # 5. 执行编码器层 -> 前向传播
    output = encoder_layer(x,mask)
    # 6. 打印结果
    print(f'编码器层输出形状: {output.shape}')      # [batch_size, seq_len, d_model] ->  [2, 4, 512]
    print(f'编码器层的输出内容: {output}')
    print(f'编码器层的输出内容: {output[:1, :2, :5]}')       # 第1个句子, 前2个单词, 每个单词的前5个特征
    print(' -.- ' * 10)

    print(f'编码器层对象: \n{encoder_layer}')

if __name__ == '__main__':
    use_encoder_layer()