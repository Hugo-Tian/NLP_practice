import fasttext

# todo 1. 定义函数，训练向量模型，保存模型
def demo01_train_save():
    # 读取数据，开始训练
    my_model = fasttext.train_unsupervised(input='./data/wh02ad',
                                           model='cbow',
                                           dim=200,
                                           epoch=10,
                                           lr=0.005,
                                           thread=20
    )
    # 保存模型
    my_model.save_model('./model/wh02_file.bin')
    print('训练完毕')

# todo 2.定义函数，实现加载模型并预测

def dm02_get_word_vector():
    model = fasttext.load_model('./model/wh02_file.bin')
    results = model.get_word_vector('cat')
    print(f'type:{type(results)}')
    print(f'shape:{results.shape}')
    print(f'results:{results}')

def dm03_get_similar_words():
    model = fasttext.load_model('./model/wh02_file.bin')
    results = model.get_nearest_neighbors('cat')

    print(f'results:{results}')

if __name__ == '__main__':
    demo01_train_save()
    dm02_get_word_vector()
    dm03_get_similar_words()