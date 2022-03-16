# »æÖÆÍ¼Ïñ

import pandas as pd
import numpy as np


def hfunc2(theta, x, y, degrees):
    temp = theta[0][0]
    place = 0
    for i in range(1, degrees+1):
        for j in range(0, i+1):
            temp += np.power(x, i-j)*np.power(y, j)*theta[0][place+1]
            place += 1
    return temp


def find_decision_boundary(theta):
    t1 = np.linspace(-1, 1, 500)
    t2 = np.linspace(-1, 1, 500)
    cordinates = [(x, y)for x in t1 for y in t2]
    x_cord, y_cord = zip(*cordinates)
    h_val = pd.DataFrame({'x1': x_cord, 'x2': y_cord})
    h_val['hval'] = hfunc2(theta, h_val['x1'], h_val['x2'], 6)

    decision = h_val[np.abs(h_val['hval']) < 3 * 10**-2]
    #decision = h_val['hval']

    return decision.x1, decision.x2
