"""梯度下降法"""

import numpy as np
from matplotlib import pyplot as plt

#收集数据，导入数据
def get_data():
    x = np.linspace(-1, 1, 100)
    y = 4*x**4 - 5*x**3 + 3*x**2 - 2*x - 1
    return x, y
x, y = get_data()

'''plt.scatter(x, y)
plt.show()'''

#创建模型/假设函数形式
W = np.ones(5)
def f(x):
    global W
    return W[0]*x**4 + W[1]*x**3 + W[2]*x**2 + W[3]*x + W[4]

'''plt.scatter(x, y)
plt.plot(x, yp)
plt.show()'''

#赋初值

#计算误差
def loss(x,y,yp):
    return np.sum((y-yp)**2)

'''yp = f(x)
l  = loss(x,y,yp)
print(l)'''

#计算梯度
def gradient(x,y,yp):
    global W
    g_w = np.zeros_like(W)
    for i in range(5):
        g_w[i] = 2*np.sum((yp-y)*x**(4-i))
    return g_w

#更新系数

#循环迭代
def train(x,y,epochs,lr):
    global W
    w_list = [W]
    l_list = []
    for epoch in range(epochs):
        yp = f(x)
        l = loss(x,y,yp)
        g_w = gradient(x,y,yp)
        W = W - lr * g_w
        w_list.append(W)
        l_list.append(l)
    return w_list,l_list



lr = 0.001
epochs = 500
w_list,l_list = train(x,y,epochs,lr)

print(l_list[-1])

plt.plot(l_list)
plt.show()

for i,W in enumerate(w_list):
    yp = f(x)
    plt.cla()
    plt.scatter(x, y)
    plt.plot(x, yp,color='red')
    plt.title(f"{i}")
    plt.draw()
    plt.pause(0.1)
plt.show()