# 计算聚类中心的函数

import numpy as np


def compute_centroids(x, idx, k):
    m, n = x.shape
    centroids = np.zeros((k, n))

    for i in range(k):
        indices = np.where(idx == i)
        centroids[i, :] = (np.sum(x[indices, :], axis=1) /
                           len(indices[0])).ravel()

    return centroids
