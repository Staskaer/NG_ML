# PCA中根据矩阵来降维数据的函数
import numpy as np


def projetc_data(x, u, k):
    # 降维函数
    u_reduce = u[:, :k]
    return np.dot(x, u_reduce)


def recover_data(z, u, k):
    # 反向转换来恢复数据
    u_reduce = u[:, :k]
    return np.dot(z, u_reduce.T)
