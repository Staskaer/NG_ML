# 正则化的梯度计算函数

import numpy as np
from sigmoid import sigmoid


def gradient_reg(theta, x, y, learning_rate):
    theta = np.matrix(theta)
    x = np.matrix(x)
    y = np.matrix(y)

    parameters = int(theta.ravel().shape[1])
    grad = np.zeros(parameters)

    error = sigmoid(x*theta.T) - y
    for i in range(parameters):
        term = np.multiply(error, x[:, i])

        if (i == 0):
            grad[i] = np.sum(term)/len(x)
        else:
            grad[i] = np.sum(term)/len(x) + \
                ((learning_rate/len(x))*theta[:, i])
    return grad
