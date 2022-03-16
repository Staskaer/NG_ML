# 获取模型数据
from scipy.io import loadmat
import matplotlib.pyplot as plt


def get_data_anomaly(draw=False):
    data = loadmat(r"ex8\ex8data1.mat")

    if draw is True:
        x = data['X']
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.scatter(x[:, 0], x[:, 1])
        plt.show()

    return data
