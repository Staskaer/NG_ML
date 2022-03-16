# 练习2

import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
from costFunction import cost_function
from plotData import plot_data
from gradient import gradient

if __name__ == "__main__":
    data = plot_data()
    data.insert(0, 'ones', 1)

    cols = data.shape[1]
    x = data.iloc[:, 0:cols-1]
    y = data.iloc[:, cols-1:cols]
    theta = np.zeros(3)

    x = np.matrix(x.values)
    y = np.matrix(y.values)

    positive = data[data['admitted'].isin([1])]
    negative = data[data['admitted'].isin([0])]

    result = opt.fmin_tnc(func=cost_function, x0=theta,
                          fprime=gradient, args=(x, y))
    # 此处是选择使用科学计算库中梯度下降算法
    # 只要给出代价函数、theta、和梯度计算函数即可
    print(result)

    plotting_x1 = np.linspace(30, 100, 100)
    plotting_h1 = (- result[0][0] - result[0][1] * plotting_x1) / result[0][2]

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot(plotting_x1, plotting_h1, 'y', label='prediction')
    ax.scatter(positive['exam1'], positive['exam2'],
               s=50, c='b', marker='o', label='Admitted')
    ax.scatter(negative['exam1'], negative['exam2'], s=50,
               c='r', marker='x', label='Not Admitted')
    ax.legend()
    ax.set_xlabel('Exam 1 Score')
    ax.set_ylabel('Exam 2 Score')
    plt.show()
