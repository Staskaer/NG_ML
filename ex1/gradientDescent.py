# Ìİ¶ÈÏÂ½µËã·¨

from computeCost import compute_cost
import numpy as np


def gradient_descent(x, y, theta, alpha, num_iters):
    temp = np.matrix(np.zeros(theta.shape))
    parameters = int(theta.ravel().shape[1])
    cost = np.zeros(num_iters)

    for i in range(num_iters):
        error = x*theta.T - y
        for j in range(parameters):
            term = np.multiply(error, x[:, j])
            temp[0, j] = theta[0, j] - ((alpha/len(x))) * np.sum(term)
        theta = temp
        cost[i] = compute_cost(x, y, theta)
    return theta, cost
