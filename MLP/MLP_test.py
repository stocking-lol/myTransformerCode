"""
目标：通过一个MLP实现图片的多分类任务
"""

import numpy as np
from torchvision import datasets, transforms
import torch
from torch.utils.data import DataLoader
from torch import nn
from matplotlib import pyplot as plt

'''
收集数据，加载数据
数据：FashionMnist数据集
图片大小：28*28灰度图,每个像素点都是0到255的数字整数。标签：10，0-9
'''

def load_data(batch_size,is_train=True,download=False):
    trans = transforms.Compose([transforms.ToTensor()])
    dataset = datasets.FashionMNIST(root='./data', train=is_train, download=download,transform=trans)
    dataloader = DataLoader(dataset=dataset,batch_size=batch_size,shuffle=True)
    return dataloader

"""
创建AI
明确AI模型的参数：MLP，隐藏层层数，每一层的数量，输入和输出
输出为每个对应标签的概率
H=RelU(XW1+B1) Y = HW2+B2
"""

num_inputs = 28*28
num_hidden1 = 256
num_outputs = 10
def MLP(X): # X:[batch,1,32,32]
    global W1,W2,B1,B2,num_inputs
    X = X.reshape((-1,num_inputs))
    H = ReLU(X@W1+B1)
    Y = H@W2+B2
    return Y

def ReLU(X):
    temp = torch.zeros_like(X)
    return torch.max(X, temp)

#权重赋初值
W1 = nn.Parameter(torch.randn(num_inputs,num_hidden1,requires_grad=True)*0.01)
W2 = nn.Parameter(torch.randn(num_hidden1,num_outputs)*0.01)
B1 = nn.Parameter(torch.zeros(num_hidden1,requires_grad=True))
B2 = nn.Parameter(torch.zeros(num_outputs,requires_grad=True))

'''dataloader = load_data(10)
for x,y in dataloader:
    y_pre = MLP(x)
    print(y_pre)
    break'''


#计算损失函数
loss = nn.CrossEntropyLoss(reduction="none")
#l = loss(y_pre,y) #l是一个向量而不是标量

#计算梯度
'''l.mean().backward()'''



#更新权重
lr = 0.1
updater = torch.optim.SGD([W1,W2,B1,B2],lr=lr)
updater.step() #移动一步。torch梯度是累加的
#每次计算完梯度清零一下
updater.zero_grad()


#循环迭代
def train(net,train_iter,updater,loss,epochs):
    l_list = []
    for epoch in range(epochs):
        for x,y in train_iter:
            y_pre = net(x)
            l = loss(y_pre,y)
            l.mean().backward()
            updater.step()
            updater.zero_grad()
        l_list.append(l.mean().item())
    return l_list

net = MLP
train_iter = load_data(32)
epochs = 10


l_list = train(net,train_iter,updater,loss,epochs)

'''plt.plot(l_list)
plt.show()'''
print(l_list[-1])

#测试集

def test(net,test_iter):
    for x,y in test_iter:
        y_pre = net(x)
        y_pre = torch.argmax(y_pre,dim=1)
        sum = 0
        for i in range(len(y_pre)):
            if y_pre[i] != y[i]:
                sum += 1
        print(sum)
        break

test_iter = load_data(100,is_train=False)
test(net,test_iter)
