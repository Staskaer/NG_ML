# �����ֵ������Э�������ĺ���
import numpy as np


def compute_basic_data(x):
    u = np.mean(x, 0)
    sigmoid = np.zeros((2, 2))
    for i in range(x.shape[0]):
        sigmoid = sigmoid + (x[i]-u).T*(x[i]-u)
    return (u.T, sigmoid)
