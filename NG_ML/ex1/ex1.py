# ���������Իع�

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from gradientDescent import gradient_descent
from plotData import plot_data

if __name__ == '__main__':
    data = plot_data()
    data.insert(0, 'ones', 1)

    cols = data.shape[1]
    x = data.iloc[:, :-1]  # ��ȥ���һ��Ϊx
    y = data.iloc[:, cols-1:cols]  # ���һ��Ϊy

    x = np.matrix(x.values)
    y = np.matrix(y.values)
    theta = np.matrix(np.array([0, 0]))

    alpha = 0.01
    iters = 1500

    g, cost = gradient_descent(x, y, theta, alpha, iters)
    # g��theta��cost�Ǵ��ۺ���
    print(g)

    # ����ԭʼ���������
    x = np.linspace(data.population.min(), data.population.max(), 100)
    f = g[0, 0]+g[0, 1]*x
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot(x, f, 'r', label='prediction')
    ax.scatter(data.population, data.profit, label='training data')
    ax.legend(loc=2)
    ax.set_xlabel('population')
    ax.set_ylabel('profit')
    ax.set_title('ml')
    plt.show()
