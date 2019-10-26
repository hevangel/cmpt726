import numpy as np
import torch
from torch import sigmoid
import random

dtype = torch.float
device = torch.device('cpu')

# Times to run
epoch = 10000

# There are 2 inputs
inputLayerSize = 2

# NN nodes
hiddenLayerSize = 3

# Only one output
outputLayerSize = 1

# Learning rate
L = 0.1

# There are 2 inputs for XOR
X = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], device=device, dtype=dtype)

# The truth table of XOR
# ANN just can learn from truly examples!!!
#     (adjust the weight and bias to make output is getting close to target by input)
Y = torch.tensor([[0], [1], [1], [0]], device=device, dtype=dtype)


Wh = torch.randn(inputLayerSize, hiddenLayerSize, device=device, dtype=dtype, requires_grad=True)
Wz = torch.randn(hiddenLayerSize, outputLayerSize, device=device, dtype=dtype, requires_grad=True)

for i in range(epoch):
    H = sigmoid(X.mm(Wh))
    Z = H.mm(Wz)
    E = torch.nn.functional.mse_loss(Y,Z)
    E.backward()

    with torch.no_grad():
        Wz -= L * Wz.grad
        Wh -= L * Wh.grad

        Wz.grad.zero_()
        Wh.grad.zero_()

print("**************** error ****************")
print(E)
print("***************** output **************")
print(Z)
print("*************** weights ***************")
print("input to hidden layer weights: ")
print(Wh)
print("hidden to output layer weights: ")
print(Wz)

# Test
fail = 0
for i in range(0,100):
    x1 = random.randint(0,1)
    x2 = random.randint(0,1)
    y_ground_truth = x1 ^ x2
    y_predict = sigmoid(torch.Tensor([[x1, x2]]).mm(Wh)).mm(Wz)
    y_predict_round = round(y_predict.item())
    print('ground truth:', y_ground_truth, 'predict:', y_predict.item(), 'round:', y_predict_round)
    if y_ground_truth != y_predict_round:
        fail += 1

print('fail rate:', fail, '/100')



