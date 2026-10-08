
import torch                                                #深度学习框架
from tensorflow.keras.preprocessing.text import Tokenizer   # 词汇映射器
from torch.utils.tensorboard import SummaryWriter           # 可视化词向量
import jieba                                                # 分词器
import torch.nn as nn                                       # 神经网络模块


# todo 1.定义函数，演示：word Embedding将文本转化为词向量
def dm01_embedding_show():
    # 1. 待处理的文本
    sentence1 = '建筑学是一门垃圾学科,知识体系是没有的,评价标准是玄学的'
    sentence2 = '我爱学习自然语言处理'
    # 两句话封装成列表
    sentences = [sentence1, sentence2]

    # 2.使用jieba进行分词，并存入词语列表
    word_list = []
    # print(f'分词结果：{word_list}')
    for sentence in sentences:
        word_list.append(jieba.lcut(sentence))

    # 3.构建词汇表，进行文本数值化（词向量）
    # 3.1 初始化词汇映射器
    my_tokenizer = Tokenizer()
    # 3.2 拟合训练数据，统计词频
    my_tokenizer.fit_on_texts(word_list)
    # print(my_tokenizer.word_index)
    # 3.3 查看去重后所有词汇列表
    my_token_list = my_tokenizer.word_index.values()
    # print(my_token_list)


    # 4. 创建词嵌入层，把文本转为词向量
    # 词嵌入层： 参1：词汇表大小，即：唯一词的个数    参2：词向量维度
    embed = nn.Embedding(len(my_token_list), 32)
    print(f'embed:{embed.weight.data}')
    print(f'embed.shape:{embed.weight.data.shape}')

    # 5. 词向量可视化
    # 5.1 创建TensorBoard写入器,将数据写入到runs目录
    my_summary = SummaryWriter('./runs')
    # 5.2 将词向量和对应的词语添加到TensorBoard中
    my_summary.add_embedding(embed.weight.data,my_token_list)
    # 5.3 关闭写入器
    my_summary.close()

    # 6.查看每个单词对应词向量
    for idx in range(len(my_tokenizer.word_index)):
        temp_vector = embed(torch.tensor(idx))
        # print(f'词向量:{temp_vector}')
        word = my_tokenizer.index_word[idx+1]
        print(f'单词:{word}，词向量：{temp_vector.detach().numpy()}')


if __name__ == '__main__':
    dm01_embedding_show()