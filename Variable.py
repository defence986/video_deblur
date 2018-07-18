import torch
from torch.autograd import Variable

# get the eggs
tensor = torch.FloatTensor([[1,2], [3,4]])
#print('\n')
#print('tensor: ', tensor)

variable = Variable(tensor, requires_grad=True)
#print(tensor)
#print('variable: \n')
#print(variable)
t_out = torch.mean(tensor*tensor)       # x^2
v_out = torch.mean(variable*variable)   # x^2
#print(t_out)
#print(v_out)    # 7.5

v_out.backward()    # ?? v_out ???????

# ??????????, ???? Variable ????????, ??????????.
# v_out = 1/4 * sum(variable*variable) ??????? v_out ????
# ??? v_out ?????, d(v_out)/d(variable) = 1/4*2*variable = variable/2

print('初始Variable的梯度\n', variable.grad)    # ?? Variable ???


#print(v_out)    # 7.5
#print(variable*variable)
print('*********************************************')
print('variable   Variable形式:\n', variable)
print('*********************************************')
print('variable.data  tensor形式:\n', variable.data)
print('*********************************************')
print('variable.data.numpy()    numpy形式:\n', variable.data.numpy())