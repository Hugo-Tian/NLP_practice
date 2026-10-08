import torch


max1=torch.randn(10,3,4)

max2=torch.randn(1,4,5)

# res1=torch.matmul(max1,max2)
# print(res1.shape)
res2=torch.bmm(max1,max2)
print(res2.shape)