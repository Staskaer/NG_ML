# 计算代价函数

import numpy as np


def compute_cost(x, y, theta):
    m = len(x)
    inner = np.power(((x*theta.T)-y), 2)
    # 乘上theta的转置，同时注意x是两列，第一列全是1
    # 相当于theta0 + theta1 * x - y
    return np.sum(inner)/(2*m)
