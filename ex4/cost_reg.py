# ���򻯵Ĵ��ۺ���������Լ����

# from get_data import get_data
# from scipy.io import loadmat
import numpy as np
from forward_function import forward


def cost_reg(theta1, theta2, input_size, hidden_size, num_labels, x, y, learning_rate):
    m = x.shape[0]
    x = np.matrix(x)
    y = np.matrix(y)

    a1, z2, a2, z3, h = forward(x, theta1, theta2)

    # ����cost
    j = 0
    for i in range(m):
        first_term = np.multiply(-y[i, :], np.log(h[i, :]))
        second_term = np.multiply(1-y[i, :], np.log(1-h[i, :]))
        j += np.sum(first_term-second_term)

    j = j/m
    # theta1 ��25*401��theta2��10*26
    # ���ڵ�һ������ƫ�ã�����Ҫ�ۼ�
    j += np.sum(float(learning_rate)/(2*m)) * \
        (np.sum(np.power(theta1[:, 1:], 2)) +
         np.sum(np.power(theta2[:, 1:], 2)))
    return j


# ���Դ���
# x, y = get_data()
# weight = loadmat(r"ex4\ex4weights.mat")
# theta1 = weight['Theta1']
# theta2 = weight['Theta2']
# j = cost_reg(theta1, theta2, 400, 25, 10, x, y, 1)
# print(j)
