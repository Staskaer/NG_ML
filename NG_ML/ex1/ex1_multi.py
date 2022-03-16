# 多个变量的线性回归

import numpy as np
from matplotlib import pyplot as plt
from normalEqn import normal_eqn
from plotDataMulti import plot_data_multi
from featureNormalize import feature_normalize
from gradientDescent import gradient_descent

if __name__ == "__main__":
    data = plot_data_multi()
    data.insert(0, 'ones', 1.0)

    cols = data.shape[1]
    x = data.iloc[:, :-1]  # 除去最后一列为x
    y = data.iloc[:, cols-1:cols]  # 最后一列为y

    x = np.matrix(x.values)
    y = np.matrix(y.values)
    theta = np.matrix(np.array([0, 0, 0]))

    # 归一化处理
    x[:, 1] = feature_normalize(x[:, 1])
    x[:, 2] = feature_normalize(x[:, 2])
    y = feature_normalize(y)

    alpha = 0.01
    iters = 1500

    g, cost = gradient_descent(x, y, theta, alpha, iters)
    print("tidu ", g)

    theta_eqn = normal_eqn(x, y)
    print("zheng ", theta_eqn)
