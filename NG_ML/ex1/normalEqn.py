# 正规方程法
import numpy as np


def normal_eqn(x, y):
    theta = np.linalg.inv(x.T@x)@x.T@y  # x.T@x == x.T.dot(x)
    return theta
