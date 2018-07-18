# -*- coding: UTF-8 -*-

import torch
from torch.autograd import Variable
import torch.nn.functional as F    # 激励函数都在这
import matplotlib.pyplot as plt


# 假数据
n_data = torch.ones(100, 2)         # 数据的基本形态
x0 = torch.normal(2*n_data, 1)      # 类型0 x data (tensor), shape=(100, 2)
y0 = torch.zeros(100)               # 类型0 y data (tensor), shape=(100, 1)
x1 = torch.normal(-2*n_data, 1)     # 类型1 x data (tensor), shape=(100, 1)
y1 = torch.ones(100)                # 类型1 y data (tensor), shape=(100, 1)
#print('n_data:\n', n_data)


# 注意 x, y 数据的数据形式是一定要像下面一样 (torch.cat 是在合并数据)
x = torch.cat((x0, x1), 0).type(torch.FloatTensor)  # FloatTensor = 32-bit floating
y = torch.cat((y0, y1), ).type(torch.LongTensor)    # LongTensor = 64-bit integer

# print('x.data.numpy():\n', x.data.numpy())
# print('x.data.numpy()[:, 0]:\n', x.data.numpy()[:, 0])
# print('x.data.numpy()[:, 1]:\n', x.data.numpy()[:, 1])



# 用 Variable 来修饰这些数据 tensor
# x, y = torch.autograd.Variable(x), Variable(y)
#x, y = Variable(x), Variable(y)

# 画图
# plt.figure(1, figsize=(8, 6))
# plt.subplot(221)
# plt.plot(x.data.numpy(), y.data.numpy(), c='red', label='regression')
# plt.ylim((-1, 5))
# plt.legend(loc='best')
#plt.scatter(x.data.numpy(), y.data.numpy())
# plt.scatter(x.data.numpy()[:, 0], x.data.numpy()[:, 1], c=y.data.numpy(), s=100, lw=0, cmap='RdYlGn')
# plt.scatter(x.data.numpy()[:, 0], x.data.numpy()[:, 1])
#
# plt.show()
#print('over..................')






# method 1
class Net(torch.nn.Module):  # 继承 torch 的 Module
    def __init__(self, n_features, n_hidden, n_output):
        super(Net, self).__init__()     # 继承 __init__ 功能
        # 定义每层用什么样的形式
        self.hidden = torch.nn.Linear(n_features, n_hidden) #隐藏层线性输出
        self.out = torch.nn.Linear(n_hidden, n_output) #输出层线性输出

    def forward(self, x): #这同时也是Module中的forward功能
        # 正向传播输入值，神经网络分析出输出值
        x = F.relu(self.hidden(x)) #激励函数（隐藏层的线性值）
        x = self.out(x)      # 输出值, 但是这个不是预测值, 预测值还需要再另外计算
        return x
#
net = Net(n_features=2, n_hidden=10, n_output=2) # 几个类别就几个 output
# print(net)


net2 = torch.nn.Sequential(
    torch.nn.Linear(2, 10),
    torch.nn.ReLU(),
    torch.nn.Linear(10, 2),
)

print(net)
print(net2)


# # optimizer是训练的工具
optimizer = torch.optim.SGD(net.parameters(), lr=0.0002) #传入net的所有参数，学习率

# 算误差的时候, 注意真实值!不是! one-hot 形式的, 而是1D Tensor, (batch,)
# 但是预测值是2D tensor (batch, n_classes)
loss_func = torch.nn.CrossEntropyLoss() #预测值和真实值的误差计算公式（均方差）
#
plt.ion()   #画图
plt.show()
#
t = 0
# for t in range(1000):
while 1:
    t = t+1
    out = net(x) #喂给net训练数据x,输出预测值
#
    loss = loss_func(out, y) #计算两者的误差
#
    optimizer.zero_grad() #清空上一步的残余更新参数值
    loss.backward() #误差反向传播，计算参数更新值
    optimizer.step() #将参数更新值施加到net的parameters上
#
    if t % 2 == 0:
        # plot and show learning process
        plt.cla()
        # 过了一道 softmax 的激励函数后的最大概率才是预测值
        prediction = torch.max(F.softmax(out), 1)[1]
        pred_y = prediction.data.numpy().squeeze()
        target_y = y.data.numpy()
        plt.scatter(x.data.numpy()[:, 0], x.data.numpy()[:, 1], c=pred_y, s=100, lw=0, cmap='RdYlGn')
        accuracy = sum(pred_y == target_y) / 200  # 预测中有多少和真实值一样
        plt.text(1.5, -4, 'Accuracy=%.2f' % accuracy, fontdict={'size': 20, 'color': 'red'})
#         plt.scatter(x.data.numpy(), y.data.numpy())
#         plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw=5)
#         # plt.text(0.5, 0, 'Loss=%.4f' % loss.data[0], fontdict={'size': 20, 'color': 'red'})
#         plt.text(0.5, 0, 'Loss=%.4f' % loss.data[0], fontdict={'size': 20, 'color': 'red'})
        plt.pause(0.2)
        if accuracy > 0.999999:
            break

plt.ioff()
plt.show()