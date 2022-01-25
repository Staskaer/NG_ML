# 读取展示数据

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_data():
    path = r'ex2\ex2data1.txt'
    data = pd.read_csv(path, header=None, names=['exam1', 'exam2', 'admitted'])

    positive = data[data['admitted'].isin([1])]
    negative = data[data['admitted'].isin([0])]

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.scatter(positive['exam1'], positive['exam2'],
               s=50, c='b', marker='o', label='admitted')
    ax.scatter(negative['exam1'], negative['exam2'], s=50,
               c='r', marker='x', label='not admitted')
    ax.legend()
    ax.set_xlabel("exam1")
    ax.set_ylabel("exam2")
    plt.show()
    return data
