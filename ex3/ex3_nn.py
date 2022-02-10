# 利用给定参数来实现前馈神经网络

from get_weight import get_weight
from sklearn.metrics import classification_report
from getData import get_data
from sigmoid import sigmoid
import numpy as np

if __name__ == "__main__":
    data = get_data()
    theta1, theta2 = get_weight()

    x = np.matrix(
        np.insert(data['X'], 0, values=np.ones(data['X'].shape[0]), axis=1))
    y = np.matrix(data['y'])

    a1 = x
    z2 = a1*theta1.T

    a2 = sigmoid(z2)
    a2 = np.insert(a2, 0, values=np.ones(a2.shape[0]), axis=1)
    z3 = a2*theta2.T

    a3 = sigmoid(z3)

    y_pred2 = np.argmax(a3, axis=1)+1
    print(classification_report(y, y_pred2))
