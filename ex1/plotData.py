# 读取并展示数据

import pandas as pd
import matplotlib.pyplot as plt


def plot_data():
    path = r"ex1\ex1data1.txt"
    data = pd.read_csv(path, header=None, names=['population', 'profit'])

    data.plot(kind='scatter', x='population', y='profit', figsize=(12, 8))
    plt.show()
    return data
