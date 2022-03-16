# �������������Ĵ��ۺ���
# ���������Իع飬��˲���Ҫsigmoid����
import numpy as np
# from sigmoid import sigmoid
# from get_data import get_data


def cost_reg(theta, x, y, learning_rate):
    # ��ԭʼ�ķ������þ�������
    x = np.matrix(x)
    y = np.matrix(y)
    theta = np.matrix(theta)
    m = x.shape[0]

    first = np.power(x*theta.T-y.T, 2)
    # �ر�Ҫע��������ľ���ά��
    second = np.power(theta[:, 1:], 2)
    return np.sum(first)/(2*m) + learning_rate/(2*m)*np.sum(second)


def cost(theta, X, y, reg):
    # ��һ�ַ���ʵ�ֵĴ��ۺ���
    # ���ڻ��ķ�������ƽ����
    m = X.shape[0]
    inner = X @ theta - y  # R(m*1)
    square_sum = inner @ inner.T
    cost = square_sum / (2 * m)

    # ��������󷨻��Ǿ��󣬺��Ԧ�0
    regularized_term = ((reg / (2 * m)) * np.power(theta[1:], 2)).sum()

    return np.sum(cost+regularized_term)
