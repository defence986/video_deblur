import torch
from torch.autograd import Variable
import torch.nn.functional as F    # 激励函数都在这
import matplotlib.pyplot as plt

x = torch.unsqueeze(torch.linspace(-1, 1, 100), dim=1)  # x data (tensor), shape=(100, 1)
y = x.pow(2) + 0.2*torch.rand(x.size())                 # noisy y data (tensor), shape=(100, 1)

# 用 Variable 来修饰这些数据 tensor
# x, y = torch.autograd.Variable(x), Variable(y)
x, y = Variable(x), Variable(y)

# 画图
# plt.figure(1, figsize=(8, 6))
# plt.subplot(221)
# plt.plot(x.data.numpy(), y.data.numpy(), c='red', label='regression')
# plt.ylim((-1, 5))
# plt.legend(loc='best')
plt.scatter(x.data.numpy(), y.data.numpy())
plt.show()
#print('over..................')

class Net(torch.nn.Module):  # 继承 torch 的 Module
    def __init__(self, n_features, n_hidden, n_output):
        super(Net, self).__init__()     # 继承 __init__ 功能
        # 定义每层用什么样的形式
        self.hidden = torch.nn.Linear(n_features, n_hidden) #隐藏层线性输出
        self.predict = torch.nn.Linear(n_hidden, n_output) #输出层线性输出

    def forward(self, x): #这同时也是Module中的forward功能
        # 正向传播输入值，神经网络分析出输出值
        x = F.relu(self.hidden(x)) #激励函数（隐藏层的线性值）
        x = self.predict(x) #输出值
        return x

net = Net(n_features=1, n_hidden=10, n_output=1)
print(net)

# optimizer是训练的工具
optimizer = torch.optim.SGD(net.parameters(), lr=0.5) #传入net的所有参数，学习率
loss_func = torch.nn.MSELoss() #预测值和真实值的误差计算公式（均方差）

plt.ion()   #画图
plt.show()

for t in range(1000):
    prediction = net(x) #喂给net训练数据x,输出预测值

    loss = loss_func(prediction, y) #计算两者的误差

    optimizer.zero_grad() #清空上一步的残余更新参数值
    loss.backward() #误差反向传播，计算参数更新值
    optimizer.step() #将参数更新值施加到net的parameters上

    if t % 5 == 0:
        # plot and show learning process
        plt.cla()
        plt.scatter(x.data.numpy(), y.data.numpy())
        plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw=5)
        # plt.text(0.5, 0, 'Loss=%.4f' % loss.data[0], fontdict={'size': 20, 'color': 'red'})
        plt.text(0.5, 0, 'Loss=%.4f' % loss.data[0], fontdict={'size': 20, 'color': 'red'})
        plt.pause(0.2)