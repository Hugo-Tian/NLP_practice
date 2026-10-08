from tensorflow.keras.preprocessing import sequence

cutlen =10
def padding(x_train):
    # 参1：待处理的文本张量
    # 参2：最大长度
    # 参3：截断策略，pre(默认，截断序列前端），post(截断后端）
    # 参4：补齐策略，pre(默认，补齐序列前端），post(补齐后端）
    return sequence.pad_sequences(x_train, maxlen=cutlen,truncating='post',padding='post')

def padding_custom(x_train):
    pass

if __name__ == '__main__':
    train = [
        [1,2,3,5,43,6,3,5,3,3,5,102,53],
        [4,7,3,5,766]
    ]
    print(padding(train))