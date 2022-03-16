# �����ݶ�

import numpy as np


def gradient(theta, x, y):
    # ����������㲻�����������ʧ
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
    # ����������
    regularized_term = theta.copy()
    regularized_term[0] = 0
    regularized_term = (reg/m)*regularized_term

    # ���㲻�����������ʧ
    inner = (x.T*(x*theta.T - y.T)).T

    return inner/m + regularized_term
