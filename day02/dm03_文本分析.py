# 导包
import jieba                        # 分词包
import seaborn as sns               # 画图包
import pandas as pd                 # 数据处理包
import matplotlib.pyplot as plt     # 画图包
from itertools import chain         # 迭代器工具
import jieba.posseg as pseg         # 词性标注包
from wordcloud import WordCloud     # 词云包

# 解决中文乱码问题
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# todo 1. 定义函数，实现：训练集和测试集的标签分布可视化统计
def dm01_label_sns_countplot():
    # 设置风格
    plt.style.use('ggplot')
    # 读取数据
    train_data = pd.read_csv('./data/train.tsv',sep='\t')
    test_data = pd.read_csv('./data/dev.tsv',sep='\t')
    # print(train_data.head())
    # print(test_data.head())
    # 采用计数柱状图可视化
    # 参1：x轴标签， 参2： 数据集， 参3： 用于分组的分类变量， 参4： 是否显示图例
    sns.countplot(x='label',data=train_data,hue='label',legend=False)
    plt.title('train_label')
    plt.show()


# todo 2. 句子长度分布的可视化统计
def dm02_len_sns_displot():
    # 获取数据
    train_data = pd.read_csv('./data/train.tsv',sep='\t')
    test_data = pd.read_csv('./data/dev.tsv',sep='\t')
    # 计算每个句子的长度
    train_data['sentence_length']=list(map(lambda x:len(x),train_data['sentence']))
    test_data['sentence_length']=list(map(lambda x:len(x),test_data['sentence']))
    # 计数柱状图绘制句子长度的分布
    sns.countplot(x='sentence_length',data=train_data)
    plt.title('训练集句子长度分布')
    plt.xticks([]) # 隐藏x轴刻度值
    # plt.show()

    # 密度曲线图
    # sns.displot(x='sentence_length',data= train_data,kde=True)
    # plt.yticks([])
    # plt.show()


    # 直方图
    sns.histplot(x='sentence_length',data=train_data,kde=True)
    plt.title('训练集句子长度分布')
    plt.show()

# todo 3.正负样本长度散点分布
def dm03_len_sns_stripplot():
    # 获取数据
    train_data = pd.read_csv('./data/train.tsv',sep='\t')
    dev_data = pd.read_csv('./data/dev.tsv',sep='\t')
    # 获取训练集数据长度列
    train_data['sentence_length']=list(map(lambda x:len(x),train_data['sentence']))
    # 获取测试集数据长度列
    dev_data['sentence_length']=list(map(lambda x:len(x),dev_data['sentence']))
    # 散点分布图， 参1： x轴标签， 参2： y轴标签，  参3：数据集
    sns.stripplot(x='label',y='sentence_length',data=train_data)
    plt.title('训练集样本长度散点分布图')
    plt.show()
    sns.stripplot(x='label',y='sentence_length',data=dev_data)
    plt.title('测试集样本长度散点分布图')
    plt.show()

# todo 4.获取去重后的词汇总数
def dm04_get_word_count():
    # 读取数据
    train_data = pd.read_csv('./data/train.tsv',sep='\t')
    test_data = pd.read_csv('./data/dev.tsv',sep='\t')
    # 统计词汇总数
    train_vocab = set(chain(*map(lambda x:jieba.lcut(x),train_data['sentence'])))
    test_vocab = set(chain(*map(lambda x:jieba.lcut(x),test_data['sentence'])))
    print(len(train_vocab))
    print(len(test_vocab))

# todo 5. 定义函数，实现：训练集和测试集的高频形容词词云
# 获取形容词列表
def get_a_list(text):
    a_list = []
    for value in pseg.lcut(text):
        # print(value.flag)
        if value.flag == 'a':
            a_list.append(value.word)
    return a_list

# 根据词列表产生词云
def get_word_cloud(keywords_list):
    wordcloud = WordCloud(
        font_path='./data/SimHei.ttf',
        max_words=100,
        background_color='white'
    )

    keywords_str = ' '.join(keywords_list)
    wordcloud.generate(keywords_str)

    plt.figure()
    plt.imshow(wordcloud,interpolation='bilinear')
    plt.axis('off')
    plt.show()

def dm05_word_cloud():
    # 场景1：处理 训练集
    # 读取数据
    train_data = pd.read_csv('./data/train.tsv',sep='\t')
    # 筛选出label = 0的样本
    p_train_data = train_data[train_data['label']==0]['sentence']
    # 对每个负样本句子，提取形容词列表，合并为一个完整的形容词词表
    p_a_train_vocab = chain(*map(get_a_list,p_train_data))
    get_word_cloud(p_a_train_vocab)

    # 场景2： 处理 测试集
    dev_data = pd.read_csv('./data/dev.tsv',sep='\t')
    p_dev_data = dev_data[dev_data['label']==0]['sentence']
    p_a_train_vocab = chain(*map(get_a_list,p_dev_data))
    get_word_cloud(p_a_train_vocab)



if __name__ == '__main__':
    # dm01_label_sns_countplot()
    # dm02_len_sns_displot()
    # dm03_len_sns_stripplot()
    # dm04_get_word_count()
    dm05_word_cloud()