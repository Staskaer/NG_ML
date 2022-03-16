# ??????
import numpy as np
from sigmoid import sigmoid


def cost(theta, x, y, learning_rate):
    theta = np.matrix(theta)
    x = np.matrix(x)
    y = np.matrix(y)
    m = len(x)

    first = np.multiply(-y, np.log(sigmoid(x*theta.T)))
    second = np.multiply((1-y), np.log(1-sigmoid(x*theta.T)))
    reg = (learning_rate/(2*m))*np.sum(np.power(theta[:, 1:theta.shape[1]], 2))
    return np.sum(first-second)/m + reg
