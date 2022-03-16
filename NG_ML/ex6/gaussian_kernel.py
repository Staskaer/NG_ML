# 实现svm使用的高斯内核
# 实际上未用到
import numpy as np


def gaussian_kernel(x1, x2, sigmoid):
    return np.exp(-(np.sum((x1-x2)**2)/(2*(sigmoid**2))))
