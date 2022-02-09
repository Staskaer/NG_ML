# 一对多的预测函数

import numpy as np
from sigmoid import sigmoid


def predict_all(x, all_theta):
    rows = x.shape[0]
    params = x.shape[1]
    num_labels = all_theta.shape[0]

    x = np.insert(x, 0, values=np.ones(rows), axis=1)
    x = np.matrix(x)
    all_theta = np.matrix(all_theta)

    h = sigmoid(x*all_theta.T)

    h_argmax = np.argmax(h, axis=1)

    h_argmax += 1
    return h_argmax
