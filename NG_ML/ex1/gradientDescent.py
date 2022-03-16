# 梯度下降算法

from computeCost import compute_cost
import numpy as np


def gradient_descent(x, y, theta, alpha, num_iters):
    temp = np.matrix(np.zeros(theta.shape))
    parameters = int(theta.ravel().shape[1])  # 参变量个数
    cost = np.zeros(num_iters)  # 每个损失值都要存储

    # 梯度下降求theta
    for i in range(num_iters):
        error = x*theta.T - y  # 用矩阵来表示每一项
        for j in range(parameters):
            term = np.multiply(error, x[:, j])  # 此处利用x的两列实现乘以系数
            temp[0, j] = theta[0, j] - ((alpha/len(x))) * np.sum(term)
        theta = temp

        cost[i] = compute_cost(x, y, theta)  # 计算当前的损失值
    return theta, cost
