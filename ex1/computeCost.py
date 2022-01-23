# 计算代价函数

import numpy as np


def compute_cost(x, y, theta):
    m = len(x)
    inner = np.power(((x*theta.T)-y), 2)
    return np.sum(inner)/(2*m)
