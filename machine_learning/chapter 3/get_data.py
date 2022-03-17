# 获取西瓜3.0alpha的数据
import matplotlib.pyplot as plt
import pandas as pd


def get_data(draw=False):
    data = pd.read_csv(r"machine_learning\chapter 3\data3_0alpha.txt",
                       header=None, names=['phi', 'rate', 'y'])
    if draw is True:
        positive = data[data['y'].isin([1])]
        negative = data[data['y'].isin([0])]

        fig, ax = plt.subplots(figsize=(12, 8))
        ax.scatter(positive['phi'], positive['rate'], s=50,
                   c='b', marker='x', label='1')
        ax.scatter(negative['phi'], negative['rate'],
                   s=50, c='r', marker='o', label='0')
        ax.legend()
        ax.set_xlabel("phi")
        ax.set_ylabel("rate")
        plt.show()

    return data


get_data()
