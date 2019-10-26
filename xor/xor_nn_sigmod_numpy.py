import numpy as np
import random

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
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])

# The truth table of XOR
# ANN just can learn from truly examples!!!
#     (adjust the weight and bias to make output is getting close to target by input)
Y = np.array([[0], [1], [1], [0]])


def sigmod(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_deriv(x):
    return x * (1 - x)


Wh = np.random.uniform(size=(inputLayerSize, hiddenLayerSize))
Wz = np.random.uniform(size=(hiddenLayerSize, outputLayerSize))

for i in range(epoch):
    H = sigmod(np.dot(X, Wh))
    Z = np.dot(H, Wz)
    E = Y - Z
    dZ = E * L
    Wz += H.T.dot(dZ)
    dH = dZ.dot(Wz.T) * sigmoid_deriv(H)
    Wh += X.T.dot(dH)

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
    y_predict = np.dot(sigmod(np.dot(np.array([x1, x2]), Wh)), Wz)[0]
    y_predict_round = round(y_predict)
    print('ground truth:', y_ground_truth, 'predict:', y_predict, 'round:', y_predict_round)
    if y_ground_truth != y_predict_round:
        fail += 1

print('fail rate:', fail, '/100')



