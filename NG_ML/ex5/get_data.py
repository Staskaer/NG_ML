# 获取数据
from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt


def get_data(show_flag=False):
    data = loadmat(r"ex5\ex5data1.mat")
    x, y, x_val, y_val, x_test, y_test = map(np.ravel, [
        data['X'], data['y'], data['Xval'], data['yval'], data['Xtest'], data['ytest']])

    if show_flag is True:
        # 当需要用到的时候才绘图
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.scatter(x, y)
        ax.set_xlabel('water_level')
        ax.set_ylabel('flow')
        plt.show()

    return (x, y, x_val, y_val, x_test, y_test)
