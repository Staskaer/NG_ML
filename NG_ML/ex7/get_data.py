# 获取k-means和PCA的数据
from scipy.io import loadmat
import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
import cv2


def get_data_k_means(draw=False):
    # k-means的数据
    data = loadmat(r"ex7\ex7data2.mat")

    if draw is True:
        data2 = pd.DataFrame(data.get('X'), columns=['X1', 'X2'])
        sb.set(context="notebook", style="white")
        sb.lmplot('X1', 'X2', data=data2, fit_reg=False)
        plt.show()

    return data['X']


def get_img(draw=False):
    # 获取将图片压缩的数据
    data = loadmat(r"ex7\bird_small.mat")

    if draw is True:
        img = cv2.imread(r"ex7\bird_small.png", cv2.IMREAD_COLOR)
        cv2.imshow("img", img)
        cv2.waitKey()

    return data


def get_data_pca(draw=False):
    # 获取pca的数据
    data = loadmat(r"ex7\ex7data1.mat")

    if draw is True:
        x = data['X']
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.scatter(x[:, 0], x[:, 1])
        plt.show()

    return data


def get_data_pca_img(draw=False):
    # 获取用于pca的图像
    data = loadmat(r"ex7\ex7faces.mat")

    if draw is True:
        x = data['X']
        pic_size = int(np.sqrt(x.shape[1]))
        grid_size = int(np.sqrt(9))

        first_n_imgs = x[:9, :]

        fig, ax = plt.subplots(
            nrows=grid_size, ncols=grid_size, sharey=True, sharex=True, figsize=(8, 8))

        for r in range(grid_size):
            for c in range(grid_size):
                ax[r, c].imshow(first_n_imgs[grid_size * r +
                                             c].reshape((pic_size, pic_size)))
                plt.xticks(np.array([]))
                plt.yticks(np.array([]))
        plt.show()
    return data['X']
