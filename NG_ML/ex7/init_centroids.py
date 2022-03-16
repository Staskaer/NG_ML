# Ëæ»ú³õÊ¼»¯

import numpy as np


def init_centroids(x, k):
    m, n = x.shape
    centroids = np.zeros((k, n))
    idx = np.random.randint(0, m, k)

    for i in range(k):
        centroids[i, :] = x[idx[i], :]
    return centroids
