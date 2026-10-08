import jieba

def one_hot():
    # 1.准备语料库
    vocabs={'周杰伦','陈奕迅','林俊杰','王力宏','伍佰'}
    # 2. 构建词汇到索引的关系
    word2index={vocab:i for i,vocab in enumerate(vocabs)}
    print(word2index)

    # 3. 对每个词进行onehot编码
    for vocab in vocabs:
        # 3.1 初始化全零列表
        zero_list=[0]*len(vocabs)
        # 3.2 获得当前词索引
        idx=word2index[vocab]
        #
        zero_list[idx]=1
        print(f'{vocab}的onehot编码为{zero_list}')

if __name__ == '__main__':
    one_hot()