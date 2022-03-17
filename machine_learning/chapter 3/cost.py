# 计算代价函数的文件

import numpy as np
from utils import compute_basic_data


def cost(positive, negative, w):
    x0 = np.matrix([negative['phi'], negative['rate']]).T
    y0 = np.matrix(negative['y']).T
    x1 = np.matrix([positive['phi'], positive['rate']]).T
    y1 = np.matrix(positive['y']).T

    u0, sigmoid0 = compute_basic_data(x0)
    u1, sigmoid1 = compute_basic_data(x1)

    first = np.power(w.T*u0-w.T*u1, 2)
    second = w.T*sigmoid0*w+w.T*sigmoid1*w
    cost = -first/second
    return cost
