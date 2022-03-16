# 获取theta的初始化值

from scipy.io import loadmat


def get_weight():
    weight = loadmat(r"ex4\ex4weights.mat")
    theta1 = weight['Theta1']
    theta2 = weight['Theta2']
    return (theta1, theta2)
