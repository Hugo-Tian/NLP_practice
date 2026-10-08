import jieba

# 精确分词模式
def dm01():
    content = '建筑学是一门垃圾专业，知识体系是没有的，评价标准是玄学的'
    result1 = jieba.cut(content, cut_all=False)
    result2 = jieba.lcut(content,cut_all=False)
    print(result1)
    print(next(result1))
    print(next(result1))
    print(result2)

# 全模式
def dm02():
    content = '建筑学是一门垃圾专业，知识体系是没有的，评价标准是玄学的'
    result1 = jieba.cut(content, cut_all=True)
    result2 = jieba.lcut(content, cut_all=True)
    print(result1)
    print(next(result1))
    print(next(result1))
    print(result2)

# 搜索引擎模式
def dm03():
    content = '建筑学是一门垃圾专业，知识体系是没有的，评价标准是玄学的'
    result1 = jieba.cut_for_search(content)
    result2 = jieba.lcut_for_search(content)
    print(result1)
    print(next(result1))
    print(next(result1))
    print(result2)

# 自定义词典
def dm04():
    content='田贵恒在香港大学学习人工智能，随后创办了贵恒教育'
    list1= jieba.lcut(content)
    print(list1)
    jieba.load_userdict('./data/dict.txt')
    list2= jieba.lcut(content)
    print(list2)


if __name__ == '__main__':
    # dm01()
    # dm02()
    # dm03()
    dm04()