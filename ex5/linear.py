# 这个文件包含了线性回归时候的代码和展示数据
# 以及绘制线性回归的学习曲线
from gradient import gradient, gradient_reg
from cost_reg import cost, cost_reg
import matplotlib.pyplot as plt
import scipy.optimize as opt
import numpy as np


def draw_linear(x, y,  pack, l=1):
    # 绘制机器学习曲线
    x_val, y_val, x_test, y_test = pack
    train_cost, cv_cost = [], []
    m = x.shape[0]
    for i in range(1, 1+m):
        res = linear_regression(x[:i, :], y[:, :i], 0)

        tc = cost_reg(res.x, x[:i, :], y[:, :i], 0)
        cv = cost_reg(res.x, x_val, y_val, 0)

        train_cost.append(tc)
        cv_cost.append(cv)

    fig, ax = plt.subplots(figsize=(12, 8))
    plt.plot(np.arange(1, m+1), train_cost, label='training cost')
    plt.plot(np.arange(1, m+1), cv_cost, label='cv cost')
    plt.legend()
    plt.show()


def linear_regression(x, y, l=1):
    # 计算模型参数
    theta = np.ones(x.shape[1])
    res = opt.minimize(fun=cost_reg, x0=theta, args=(
        x, y, l), method='TNC', jac=gradient_reg, options={'disp': True})
    return res


def linear(x, y, theta,  draw_flag=False, pack=None):
    # 线性模型入口
    res = linear_regression(x, y, 1)
    final_theta = res.x

    # 绘图
    if draw_flag is True:
        b = final_theta[0]  # intercept
        m = final_theta[1]  # slope

        x_ = np.array(x[:, 1])
        y_ = np.array(y.T)
        fig, ax = plt.subplots(figsize=(12, 8))
        plt.scatter(x_, y_, c='r', label="Training data")
        plt.plot(x_, x_*m + b, c='b', label="Prediction")
        ax.set_xlabel('water_level')
        ax.set_ylabel('flow')
        ax.legend()
        plt.show()

    # 调用绘制学习曲线的函数
    draw_linear(x, y,  pack, 1)

    return final_theta
