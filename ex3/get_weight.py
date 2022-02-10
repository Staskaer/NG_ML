# 获取神经网络的权重

from scipy.io import loadmat


def get_weight():
    weight = loadmat(r"ex3\ex3weights.mat")
    theta1, theta2 = weight['Theta1'], weight['Theta2']

    return theta1, theta2
