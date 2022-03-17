# 线性判别分析


import numpy as np
from cost import cost
from get_data import get_data
from compute_w import compute_w
import matplotlib.pyplot as plt


if __name__ == "__main__":
    data = get_data()

    positive = data[data['y'].isin([1])]
    negative = data[data['y'].isin([0])]
    x1 = np.matrix(data['phi']).T
    x2 = np.matrix(data['rate']).T

    y = np.matrix(data['y']).T
    x = np.hstack((x1, x2))
    # w = np.matrix([1, 1]).T

    w = compute_w(positive, negative)

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.scatter(positive['phi'], positive['rate'], s=50,
               c='b', marker='x', label='1')
    ax.scatter(negative['phi'], negative['rate'],
               s=50, c='r', marker='o', label='0')
    ax.legend()
    ax.set_xlabel("phi")
    ax.set_ylabel("rate")
    plt.plot([0, -1*w[0]], [0, -1*w[1]], linewidth=1)
    plt.show()
