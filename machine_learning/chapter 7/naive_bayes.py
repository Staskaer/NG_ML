# 朴素贝叶斯
import numpy as np


def naive_bayes(X, y, X_test):
    rows, cols = X_test.shape
    result = np.zeros((X_test.shape[0], 1))
    p = np.zeros((X_test.shape[0], 1))
    for i in range(rows):
        # 待验证的每个样本
        x_temp = X_test[i]
        # 经过拉普拉斯修正后的P(c)
        P_0 = (len(y[y == 0].T)+1)/(len(X)+2)
        P_1 = (len(y[y == 1].T)+1)/(len(X)+2)
        for j in range(cols):
            attr = x_temp[0, j]  # 遍历每个属性
            N_i = len(set(X[:, j].T.tolist()[0]))  # 先将列向量转换成行向量转换成列表然后取首去重
            d0 = len(y[y == 0].T)
            d1 = len(y[y == 1].T)
            # X0，X1是0、1类别对应的样本
            X0 = X[np.matrix(y == 0).tolist()[0]]
            X1 = X[np.matrix(y == 1).tolist()[0]]
            # 这两是0、1样本中属性i为attr的样本数目
            D_0_x = np.sum(X0[:, j].T == attr)
            D_1_x = np.sum(X1[:, j].T == attr)
            # 这两是在0、1下出现xi=attr的概率
            p_xi_0 = (D_0_x+1)/(d0+N_i)
            p_xi_1 = (D_1_x+1)/(d1+N_i)
            P_0 *= p_xi_0
            P_1 *= p_xi_1
        if P_0 > P_1:
            result[i, 0] = 0
            p[i, 0] = P_0
        else:
            result[i, 0] = 1
            p[i, 0] = P_1

    return result, p
