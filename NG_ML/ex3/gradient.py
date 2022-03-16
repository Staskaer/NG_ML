# 实现梯度计算

# 这是多分类器，是把所有像素点看作是参数，然后用同等数目的参数
# 和之前的gradient基本没差别，但是此处使用矩阵进行正则化表示了
# 形式更加简洁
import numpy as np
from sigmoid import sigmoid


def gradient(theta, x, y, learning_rate):
    theta = np.matrix(theta)
    x = np.matrix(x)
    y = np.matrix(y)

    #parameters = int(theta.ravel().shape[1])
    error = sigmoid(x*theta.T) - y

    # grad = 1/m*x.T*(h(x)-y) + lambda/m*theta
    grad = ((x.T*error)/len(x)).T + ((learning_rate/(len(x)))*theta)

    # 偏置项不需要约束
    grad[0, 0] = np.sum(np.multiply(error, x[:, 0]))/len(x)

    return np.array(grad).ravel()
