# ¹éÒ»»¯º¯Êı

import numpy as np


def feature_normalize(x):
    u = np.median(x)
    s = x.max()-x.min()
    return (x - u)/s
