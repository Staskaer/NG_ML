# ¶ÁÈ¡Êý¾Ý

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_data_boundary():
    path = r"ex2\ex2data2.txt"
    data = pd.read_csv(path, header=None, names=['test1', 'test2', 'accepted'])

    positive = data[data['accepted'].isin([1])]
    negative = data[data['accepted'].isin([0])]

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.scatter(positive['test1'], positive['test2'],
               s=50, c='b', marker='o', label='accepted')
    ax.scatter(negative['test1'], negative['test2'], s=50,
               c='r', marker='x', label='not accepted')
    ax.legend()
    ax.set_xlabel('test1 score')
    ax.set_ylabel('test2 score')
    plt.show()

    return data, positive, negative
