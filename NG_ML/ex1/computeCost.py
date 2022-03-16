# ������ۺ���

import numpy as np


def compute_cost(x, y, theta):
    m = len(x)
    inner = np.power(((x*theta.T)-y), 2)
    # ����theta��ת�ã�ͬʱע��x�����У���һ��ȫ��1
    # �൱��theta0 + theta1 * x - y
    return np.sum(inner)/(2*m)
