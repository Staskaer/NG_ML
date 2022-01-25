# ´ú¼Ûº¯Êý

from sigmoid import sigmoid
import numpy as np


def cost_function(theta, x, y):
    theta = np.matrix(theta)
    x = np.matrix(x)
    y = np.matrix(y)

    first = np.multiply(-y, np.log(sigmoid(x*theta.T)))
    second = np.multiply((1-y), np.log(1-sigmoid(x*theta.T)))
    return (np.sum(first-second)/len(x))
