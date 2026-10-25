import torch

# 1. 生成1个随机整数张量, 取值范围: [1, 10), shape: [3, 5]
input = torch.randint(0, 10, (3,5))


# 2. 定义掩码张量mask
mask = torch.tensor([
    [1,2,4,0,0],
    [1,5,0,0,0],
    [1,0,0,0,0]
])


# 3. 处理mask, 把非0的值都改成1
mask[mask!=0]=1

# 4. 用 masked_fill()函数处理input, 进行掩码操作.
result = torch.masked_fill(input, mask==0,1e-9)
print(result)