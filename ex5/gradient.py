# 计算梯度

import numpy as np


def gradient(theta, x, y):
    # 这个函数计算不带正则项的损失
    x = np.matrix(x)
    y = np.matrix(y)
    theta = np.matrix(theta)

    m = x.shape[0]
    inner = x.T*(x*theta.T - y)
    return inner/m


def gradient_reg(theta, x, y, reg):
    x = np.matrix(x)
    y = np.matrix(y)
    theta = np.matrix(theta)

    m = x.shape[0]
    # 计算正则项
    regularized_term = theta.copy()
    regularized_term[0] = 0
    regularized_term = (reg/m)*regularized_term

    # 计算不带正则项的损失
    inner = (x.T*(x*theta.T - y.T)).T

    return inner/m + regularized_term
