# 通过这个程序来学习学习曲线、过/欠拟合之间的联系

# 每个文件内都有带正则化和不带正则化的两种形式
from gradient import gradient, gradient_reg
from cost_reg import cost, cost_reg
from ploy_features import ploy_features_main
from linear import linear
from get_data import get_data
import numpy as np

if __name__ == "__main__":
    # 获取数据
    x, y, x_val, y_val, x_test, y_test = get_data()
    x, x_val, x_test = [np.insert(x.reshape(x.shape[0], 1), 0, np.ones(
        x.shape[0]), axis=1) for x in (x, x_val, x_test)]

    x = np.matrix(x)
    y = np.matrix(y)
    theta = np.ones(x.shape[1])

    # 首先进行线性模型的拟合操作
    # 可以观测到机器学习曲线非常的接近，说明处于欠拟合状态

    #linear(x, y, theta, False, (x_val, y_val, x_test, y_test))

    # 然后观察多项式拟合

    # 这个函数内有两种显示方式
    # type = 0是显示m-cost图像
    # type = 1是显示lambda-cost图像
    ploy_features_main(reg=1000, power=10, type=0)
