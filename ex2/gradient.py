# 此文件实现了计算偏导数（梯度）的函数，而没有实现下降
import numpy as np
from sigmoid import sigmoid


def gradient(theta, x, y):
    theta = np.matrix(theta)
    x = np.matrix(x)
    y = np.matrix(y)

    parameters = int(theta.ravel().shape[1])
    grad = np.zeros(parameters)
    error = sigmoid(x*theta.T) - y
    for i in range(parameters):
        term = np.multiply(error, x[:, i])
        grad[i] = np.sum(term)/len(x)

    return grad
