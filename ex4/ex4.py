# 实现反向传播的神经网络

import numpy as np
from scipy.optimize import minimize
from random_init import random_init
from back_prop_reg import backpropReg
from get_data import get_data
from forward_function import forward
from sklearn.metrics import classification_report


if __name__ == '__main__':
    input_size = 400
    hidden_size = 25
    num_labels = 10
    learning_rate = 1
    x, y, y_raw = get_data()
    x = np.matrix(x)
    y = np.matrix(y)

    params = random_init(hidden_size, input_size, num_labels)
    fmin = minimize(fun=backpropReg, x0=(params), args=(input_size, hidden_size, num_labels, x, y, learning_rate),
                    method='TNC', jac=True, options={'maxiter': 250})

    print(fmin)

    # 计算准确率
    X = np.matrix(x)
    thetafinal1 = np.matrix(np.reshape(
        fmin.x[:hidden_size * (input_size + 1)], (hidden_size, (input_size + 1))))
    thetafinal2 = np.matrix(np.reshape(
        fmin.x[hidden_size * (input_size + 1):], (num_labels, (hidden_size + 1))))

    a1, z2, a2, z3, h = forward(X, thetafinal1, thetafinal2)
    y_pred = np.array(np.argmax(h, axis=1) + 1)
    print(classification_report(y_raw, y_pred))
