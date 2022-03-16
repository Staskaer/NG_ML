# Ç°Ïò´«²¥

import numpy as np
from sigmoid import sigmoid


def forward(x, theta1, theta2):
    m = x.shape[0]

    a1 = np.insert(x, 0, values=np.ones(m), axis=1)
    z2 = a1*theta1.T
    a2 = np.insert(sigmoid(z2), 0, values=np.ones(m), axis=1)
    z3 = a2*theta2.T
    h = sigmoid(z3)
    return (a1, z2, a2, z3, h)
