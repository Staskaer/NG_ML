# PCA�и��ݾ�������ά���ݵĺ���
import numpy as np


def projetc_data(x, u, k):
    # ��ά����
    u_reduce = u[:, :k]
    return np.dot(x, u_reduce)


def recover_data(z, u, k):
    # ����ת�����ָ�����
    u_reduce = u[:, :k]
    return np.dot(z, u_reduce.T)
