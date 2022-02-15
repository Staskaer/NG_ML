# 支持向量机

import numpy as np
from get_date import get_data_1, get_data2, get_data3
from find_boundary import find_decision_boundary
from sklearn import svm
import matplotlib.pyplot as plt


def svm_linear(C=1):
    # 这个函数测试以下svm使用线性核
    svc = svm.LinearSVC(C=C, loss='hinge', max_iter=1000)
    data = get_data_1()
    # 训练数据
    svc.fit(data[['X1', 'X2']], data['y'])
    # print(svc.score(data[['X1', 'X2']], data['y']))

    # 绘图
    # 先画出数据
    positive = data[data['y'].isin([1])]
    negative = data[data['y'].isin([0])]
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.scatter(positive['X1'], positive['X2'],
               s=50, marker='x', label='positive')
    ax.scatter(negative['X1'], negative['X2'],
               s=50, marker='o', label='negative')
    # 绘制边界
    x1, x2 = find_decision_boundary(svc, 0, 4, 1.5, 5, 2*10**-3)
    ax.scatter(x1, x2, s=10, c='r', label='Boundary')
    ax.set_title("svm(c={})".format(C))
    # 显示
    ax.legend()
    plt.show()


def svm_gaussian(C=1):
    # 使用高斯核的svm
    svc = svm.SVC(C=C, kernel='rbf', gamma=10, probability=True)
    data = get_data2()
    svc.fit(data[['X1', 'X2']], data['y'])

    # print(svc.score(data[['X1', 'X2']], data['y']))

    # 绘图
    # 先画出数据
    positive = data[data['y'].isin([1])]
    negative = data[data['y'].isin([0])]
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.scatter(positive['X1'], positive['X2'],
               s=50, marker='x', label='positive')
    ax.scatter(negative['X1'], negative['X2'],
               s=50, marker='o', label='negative')
    # 绘制边界
    x1, x2 = find_decision_boundary(svc, 0, 1, 0.4, 1, 0.01)
    ax.scatter(x1, x2, s=10, c='r', label='Boundary')
    ax.set_title("svm(c={})".format(C))
    # 显示
    ax.legend()
    plt.show()


def svm_gaussian_find_c_delta():
    # 寻找最优的c和delta，并绘制其拟合曲线
    x, y, x_val, y_val, data = get_data3()
    c_values = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100]
    gamma_values = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100]
    best_score = 0
    best_params = {'c': None, 'gamma': None}

    for c in c_values:
        for gamma in gamma_values:
            svc = svm.SVC(C=c, gamma=gamma, kernel='rbf')
            svc.fit(x, y)
            score = svc.score(x_val, y_val)

            if score > best_score:
                best_score = score
                best_params['c'] = c
                best_params['gamma'] = gamma

    print("best_c : {}, best_gamma : {}".format(
        best_params['c'], best_params['gamma']))

    # 绘图
    fig, ax = plt.subplots(figsize=(12, 8))
    positive = data[data['y'].isin([1])]
    negative = data[data['y'].isin([0])]
    ax.scatter(positive['X1'], positive['X2'],
               s=50, marker='x', label='positive')
    ax.scatter(negative['X1'], negative['X2'],
               s=50, marker='o', label='negative')
    # 获取训练数据
    svc = svm.SVC(C=best_params['c'], gamma=best_params['gamma'])
    svc.fit(x, y)
    x1, x2 = find_decision_boundary(svc, -0.6, 0.3, -0.7, 0.6, 0.005)
    ax.scatter(x1, x2, s=10)
    plt.show()


if __name__ == "__main__":
    # 先观察线性核的c的不同值的分类情况
    # svm_linear(1)

    # 然后观察高斯核的c的不同值的拟合情况
    # svm_gaussian(100)

    # 然后对于第三个数据集寻找最优的c和delta
    # 候选数值为[0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100]
    svm_gaussian_find_c_delta()
