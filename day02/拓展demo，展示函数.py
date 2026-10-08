from itertools import chain
import jieba


def dm02_chain():
    list1,list2=[1,2,3],[3,4,5]


    # chain()函数把两个list合并成1个，创建时不会立即遍历底层可迭代对象，
    # 只有在实际迭代时才会逐个获取元素
    result = chain(list1,list2)
    print(result)
    print(list(result))

    list3=['今天天很好','今天天很热']
    tmp = map(lambda x:jieba.lcut(x),list3)
    result = set(chain(*tmp))
    print(result)

# zip()函数功能，合并迭代对象
def dm03_zip():

    list1,list2=[1,22,333,10,20,30],[3,4,5]

    result = zip(list1,list2)
    print(result)
    print(list(result))

    list3=[[1,2,3],[3,4,5]]

    result1 = zip(list3)
    result2 = zip(*list3)
    print(list(result1))
    print(list(result2))



if __name__ == '__main__':
    # dm02_chain()
    dm03_zip()