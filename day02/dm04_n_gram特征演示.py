ngram_range=2

def create_ngram(input_list):
    # 通过滑动窗口，获取n_gram特征
    sliced_lists=[input_list[i:] for i in range(ngram_range)]
    # 用zip()函数，对切片列表进行组合
    ngram_tuples=zip(*sliced_lists)
    # 用set去重
    return set(ngram_tuples)

if __name__ == '__main__':
    input_list=[1,2,3,5,31,98]
    result = create_ngram(input_list)
    print(result)