# 练习7

from get_data import get_data_k_means, get_data_pca_img, get_img, get_data_pca
from find_closest_centroids import find_closest_centroids
from init_centroids import init_centroids
from compute_centroids import compute_centroids
from reduce_date import recover_data, projetc_data
import numpy as np
import cv2
import matplotlib.pyplot as plt


def run_k_means(max_iter=20):
    # k-means算法实现聚类
    x = get_data_k_means()
    centroids = init_centroids(x, 3)

    m, n = x.shape
    idx = idx_ = np.zeros(m)

    for i in range(max_iter):
        idx = find_closest_centroids(x, centroids=centroids)
        centroids = compute_centroids(x, idx, 3)

        # 绘图
        cluster1 = x[np.where(idx == 0)[0], :]
        cluster2 = x[np.where(idx == 1)[0], :]
        cluster3 = x[np.where(idx == 2)[0], :]
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.scatter(cluster1[:, 0], cluster1[:, 1],
                   s=30, color='r', label='Cluster 1')
        ax.scatter(cluster2[:, 0], cluster2[:, 1],
                   s=30, color='g', label='Cluster 2')
        ax.scatter(cluster3[:, 0], cluster3[:, 1],
                   s=30, color='b', label='Cluster 3')
        ax.legend()
        plt.show()

        # 收敛时停止
        if (idx == idx_).all():
            break
        idx_ = idx


def run_k_means_pic(max_iter=10):
    # 用k-means算法实现图片压缩
    # 用聚类找到最具代表性的少数颜色，
    # 并将原始的24位图像映射到低维的颜色空间
    a = get_img()['A']
    a = a/255  # 归一化
    x = np.reshape(a, (a.shape[0]*a.shape[1], a.shape[2]))

    # 执行k-means算法
    # 压缩到的位数由此处确定
    centroids = init_centroids(x, 16)
    m, n = x.shape
    idx = np.zeros(m)

    for i in range(max_iter):
        idx = find_closest_centroids(x, centroids=centroids)
        centroids = compute_centroids(x, idx, 3)

    x_recovered = centroids[idx.astype(int), :]
    x_recovered = np.reshape(x_recovered, (a.shape[0], a.shape[1], a.shape[2]))
    plt.imshow(x_recovered)
    plt.show()


def PCA():
    # 主成分分析
    data = get_data_pca()
    x = data['X']
    x = (x-x.mean())/x.std()  # 归一化

    # 计算协方差
    x = np.matrix(x)
    cov = (x.T*x)/x.shape[0]

    # svd分解
    U, S, V = np.linalg.svd(cov)

    # 先对原始数据降维再升维
    z = projetc_data(x, U, 1)
    x_rec = recover_data(z, U, 1)
    # 绘图
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.scatter(list(x_rec[:, 0]), list(x_rec[:, 1]))
    plt.show()


def PCA_img():
    # 将pca用于图像数据的降维
    x = get_data_pca_img()
    # 选一张图像来pca降维
    face = np.reshape(x[3, :], (32, 32))
    plt.imshow(face)
    plt.show()
    # 用opencv显示效果不好
    # face_ = cv2.resize(face, (400, 400))
    # cv2.imshow("raw", face_)

    x = (x-x.mean())/x.std()  # 归一化
    # 计算协方差
    x = np.matrix(x)
    cov = (x.T*x)/x.shape[0]
    # svd分解
    U, S, V = np.linalg.svd(cov)

    z = projetc_data(x, U, 100)
    x_rec = recover_data(z, U, 100)

    face_Rec = np.reshape(x_rec[3, :], (32, 32))
    plt.imshow(face_Rec)
    plt.show()
    # face_r = cv2.resize(face_Rec, (400, 400))
    # cv2.imshow("rec", face_r)
    # cv2.waitKey(-1)


if __name__ == "__main__":
    PCA_img()
