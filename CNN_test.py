"""
View more, visit my tutorial page: https://morvanzhou.github.io/tutorials/
My Youtube Channel: https://www.youtube.com/user/MorvanZhou
Dependencies:
torch: 0.4
torchvision
matplotlib
"""
# library
# standard library
import os

# third-party library
import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision
import matplotlib.pyplot as plt

# torch.manual_seed(1)    # reproducible

# Hyper Parameters
EPOCH = 1               # train the training data n times, to save time, we just train 1 epoch
BATCH_SIZE = 50
LR = 0.001              # learning rate
DOWNLOAD_MNIST = False


# Mnist digits dataset
if not(os.path.exists('./mnist/')) or not os.listdir('./mnist/'):
    # not mnist dir or mnist is empyt dir
    DOWNLOAD_MNIST = True

train_data = torchvision.datasets.MNIST(
    root='./mnist/',
    train=True,                                     # this is training data
    transform=torchvision.transforms.ToTensor(),    # Converts a PIL.Image or numpy.ndarray to
                                                    # torch.FloatTensor of shape (C x H x W) and normalize in the range [0.0, 1.0]
    download=DOWNLOAD_MNIST,
)

# plot one example
# print('**********************************************************************')
# print('train_data.train_data.size():\n', train_data.train_data.size())                 # (60000, 28, 28)
# print('\ntrain_data:\n', train_data)
# print('\ntrain_data.train_labels.size():\n', train_data.train_labels.size())               # (60000)
# print('\ntrain_data.train_labels:\n', train_data.train_labels)               # (60000)


# print('**********************************************************************')
# print('train_data.train_data:\n', train_data.train_data)
print('train_data.train_data[0].numpy():\n', train_data.train_data[0].numpy())
print('train_data.train_data[0].size():\n', train_data.train_data[0].size())


plt.imshow(train_data.train_data[0].numpy(), cmap='gray')
plt.title('%i' % train_data.train_labels[0])
plt.show()