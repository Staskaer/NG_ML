# 计算带有正则化项的代价函数
# 由于是线性回归，因此不需要sigmoid函数
import numpy as np
# from sigmoid import sigmoid
# from get_data import get_data


def cost_reg(theta, x, y, learning_rate):
    # 最原始的方法，用矩阵来求
    x = np.matrix(x)
    y = np.matrix(y)
    theta = np.matrix(theta)
    m = x.shape[0]

    first = np.power(x*theta.T-y.T, 2)
    # 特别要注意这里面的矩阵维度
    second = np.power(theta[:, 1:], 2)
    return np.sum(first)/(2*m) + learning_rate/(2*m)*np.sum(second)


def cost(theta, X, y, reg):
    # 另一种方法实现的代价函数
    # 用内积的方法来求平方和
    m = X.shape[0]
    inner = X @ theta - y  # R(m*1)
    square_sum = inner @ inner.T
    cost = square_sum / (2 * m)

    # 正则项的求法还是矩阵，忽略θ0
    regularized_term = ((reg / (2 * m)) * np.power(theta[1:], 2)).sum()

    return np.sum(cost+regularized_term)
