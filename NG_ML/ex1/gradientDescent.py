# �ݶ��½��㷨

from computeCost import compute_cost
import numpy as np


def gradient_descent(x, y, theta, alpha, num_iters):
    temp = np.matrix(np.zeros(theta.shape))
    parameters = int(theta.ravel().shape[1])  # �α�������
    cost = np.zeros(num_iters)  # ÿ����ʧֵ��Ҫ�洢

    # �ݶ��½���theta
    for i in range(num_iters):
        error = x*theta.T - y  # �þ�������ʾÿһ��
        for j in range(parameters):
            term = np.multiply(error, x[:, j])  # �˴�����x������ʵ�ֳ���ϵ��
            temp[0, j] = theta[0, j] - ((alpha/len(x))) * np.sum(term)
        theta = temp

        cost[i] = compute_cost(x, y, theta)  # ���㵱ǰ����ʧֵ
    return theta, cost
