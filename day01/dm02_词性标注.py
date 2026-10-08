import jieba.posseg as pseg

content = '我爱济宁邹城市夹饼和川味面条'

result = pseg.cut(content)
print(result)
for word, flag in result:
    print(f'{word}\t{flag}')

