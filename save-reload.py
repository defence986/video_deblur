import torch
from torch.autograd import Variable
import torch.nn.functional as F    # 激励函数都在这
import matplotlib.pyplot as plt

x = torch.unsqueeze(torch.linspace(-1, 1, 100), dim=1)  # x data (tensor), shape=(100, 1)
y = x.pow(2) + 0.2*torch.rand(x.size())                 # noisy y data (tensor), shape=(100, 1)

x, y = Variable(x, requires_grad=False), Variable(y, requires_grad=False)

#
# class Net(torch.nn.Module):  # 继承 torch 的 Module
#     def __init__(self, n_features, n_hidden, n_output):
#         super(Net, self).__init__()     # 继承 __init__ 功能
#         # 定义每层用什么样的形式
#         self.hidden = torch.nn.Linear(n_features, n_hidden) #隐藏层线性输出
#         self.predict = torch.nn.Linear(n_hidden, n_output) #输出层线性输出
#
#     def forward(self, x): #这同时也是Module中的forward功能
#         # 正向传播输入值，神经网络分析出输出值
#         x = F.relu(self.hidden(x)) #激励函数（隐藏层的线性值）
#         x = self.predict(x) #输出值
#         return x
#
# net1 = Net(n_features=1, n_hidden=10, n_output=1)


def save():
    # 建网络
    net1 = torch.nn.Sequential(
        torch.nn.Linear(1, 10),
        torch.nn.ReLU(),
        torch.nn.Linear(10, 1)
    )
    optimizer = torch.optim.SGD(net1.parameters(), lr=0.5)
    loss_func = torch.nn.MSELoss()

    # 训练
    for t in range(100):
        prediction = net1(x)
        loss = loss_func(prediction, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


# plot result
    plt.figure(1, figsize=(10, 3))
    plt.subplot(131)
    plt.title('Net1')
    plt.scatter(x.data.numpy(), y.data.numpy())
    plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw=5)
    # plt.show()

    torch.save(net1, 'net.pkl')  # 保存整个网络
    torch.save(net1.state_dict(), 'net_params.pkl')   # 只保存网络中的参数 (速度快, 占内存少)



def restore_net():
    # restore entire net1 to net2
    net2 = torch.load('net.pkl')
    prediction = net2(x)

    # plot result
    plt.subplot(132)
    plt.title('Net2')
    plt.scatter(x.data.numpy(), y.data.numpy())
    plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw=5)


def restore_params():
    # 新建 net3
    net3 = torch.nn.Sequential(
        torch.nn.Linear(1, 10),
        torch.nn.ReLU(),
        torch.nn.Linear(10, 1)
    )

    # 将保存的参数复制到 net3
    net3.load_state_dict(torch.load('net_params.pkl'))
    prediction = net3(x)
    # print(net3)
    # plot result
    plt.subplot(133)
    plt.title('Net3')
    plt.scatter(x.data.numpy(), y.data.numpy())
    plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw=5)
    plt.show()


# 保存 net1 (1\. 整个网络, 2\. 只有参数)
save()

# 提取整个网络
restore_net()

# 提取网络参数, 复制到新网络
restore_params()
