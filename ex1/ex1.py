# 单变量线性回归

import numpy as np
import pandas as pd
from gradientDescent import gradient_descent
from plotData import plot_data

if __name__ == '__main__':
    data = plot_data()
    data.insert(0, 'ones', 1)

    cols = data.shape[1]
    x = data.iloc[:, :-1]  # 除去最后一列为x
    y = data.iloc[:, cols-1:cols]  # 最后一列为y

    x = np.matrix(x.values)
    y = np.matrix(y.values)
    theta = np.matrix(np.array([0, 0]))

    alpha = 0.01
    iters = 1500

    g, cost = gradient_descent(x, y, theta, alpha, iters)
    # g是theta，cost是代价函数
    print(g)
