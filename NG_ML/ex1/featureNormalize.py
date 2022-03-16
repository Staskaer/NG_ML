# ???

import numpy as np


def feature_normalize(x):
    u = x.mean()
    s = x.max()-x.min()
    return ((x-u)/s)
