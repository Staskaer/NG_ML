# 正则化逻辑回归

import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt
from plotDecisionBoundary import plot_data_boundary
from costFunctionReg import cost_function_reg
from drawingReg import find_decision_boundary
from gradientReg import gradient_reg


if __name__ == "__main__":
    degrees = 6

    data, positive, negative = plot_data_boundary()
    x1 = data['test1']
    x2 = data['test2']

    data.insert(3, 'ones', 1)

    for i in range(1, degrees+1):
        for j in range(0, i+1):
            data['F'+str(i-j)+str(j)] = np.power(x1, i-j) * np.power(x2, j)
    # 此处是选取了六个度来组成以下向量
    # [1,x1,x2,x1^2,x1x2,x2^2,x1^3......x1x2^5,x2^6]
    # 这些特征量来作为每组数据的特征，
    # 也就相当于每组数据由原来的两个特征量变成28个特征量
    # 但由于特征量太多，所以使用正则化约束
    data.drop('test1', axis=1, inplace=True)
    data.drop('test2', axis=1, inplace=True)

    print(data.head())

    cols = data.shape[1]
    x = data.iloc[:, 1:cols]
    y = data.iloc[:, 0:1]
    theta = np.zeros(cols-1)

    x = np.array(x.values)
    y = np.array(y.values)

    learning_rate = 0.5

    result = opt.fmin_tnc(func=cost_function_reg, x0=theta,
                          fprime=gradient_reg, args=(x, y, learning_rate))
    print(result)

    # 绘制部分
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.scatter(positive['test1'], positive['test2'],
               s=50, c='b', marker='o', label='accepted')
    ax.scatter(negative['test1'], negative['test2'], s=50,
               c='r', marker='x', label='not accepted')
    ax.set_xlabel('test1 score')
    ax.set_ylabel('test2 score')

    x, y = find_decision_boundary(result)
    plt.scatter(x, y, c='y', s=10, label='prediction')
    ax.legend()
    plt.show()
