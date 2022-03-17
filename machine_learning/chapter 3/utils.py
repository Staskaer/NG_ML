# 计算均值向量，协方差矩阵的函数
import numpy as np


def compute_basic_data(x):
    u = np.mean(x, 0)
    sigmoid = np.zeros((2, 2))
    for i in range(x.shape[0]):
        sigmoid = sigmoid + (x[i]-u).T*(x[i]-u)
    return (u.T, sigmoid)
