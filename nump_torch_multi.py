import torch
import numpy as np

data = [[1,2], [3,4]]
tensor = torch.FloatTensor(data) #tensor
# correct method
print(
    '\nmatrix multiplication (matmul)',
    '\nnumpy: ', np.matmul(data, data),     # [[7, 10], [15, 22]]
    '\ntorch: ', torch.mm(tensor, tensor)   # [[7, 10], [15, 22]]
)

# !!!!  
data = np.array(data)
print(
    '\nmatrix multiplication (dot)',
    '\nnumpy: ', data.dot(data),        # [[7, 10], [15, 22]] 
    #'\ntorch: ', tensor.dot(tensor)     # torch translate into  [1,2,3,4].dot([1,2,3,4) = 30.0
)
