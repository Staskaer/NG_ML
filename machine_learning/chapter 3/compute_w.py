# º∆À„w
import numpy as np
from utils import compute_basic_data


def compute_w(positive, negative):
    x0 = np.matrix([negative['phi'], negative['rate']]).T
    y0 = np.matrix(negative['y']).T
    x1 = np.matrix([positive['phi'], positive['rate']]).T
    y1 = np.matrix(positive['y']).T

    u0, sigmoid0 = compute_basic_data(x0)
    u1, sigmoid1 = compute_basic_data(x1)

    sw = sigmoid0+sigmoid1
    w = np.linalg.pinv(sw)*(u0-u1)
    return w
