# ���򻯵ķ��򴫲�
import numpy as np
from forward_function import forward
from sigmoid import sigmoid_gradient
from cost_reg import cost_reg


def backpropReg(params, input_size, hidden_size, num_labels, X, y, learning_rate):
    m = X.shape[0]
    X = np.matrix(X)
    y = np.matrix(y)

    # ��params����ȡtheta
    theta1 = np.matrix(np.reshape(
        params[:hidden_size * (input_size + 1)], (hidden_size, (input_size + 1))))
    theta2 = np.matrix(np.reshape(
        params[hidden_size * (input_size + 1):], (num_labels, (hidden_size + 1))))

    # ǰ�򴫲���¼����ֵ
    a1, z2, a2, z3, h = forward(X, theta1, theta2)

    # ��ʼ��
    J = 0
    delta1 = np.zeros(theta1.shape)  # (25, 401)
    delta2 = np.zeros(theta2.shape)  # (10, 26)

    # ������ʧ,��ͬ��cost����
    for i in range(m):
        first_term = np.multiply(-y[i, :], np.log(h[i, :]))
        second_term = np.multiply((1 - y[i, :]), np.log(1 - h[i, :]))
        J += np.sum(first_term - second_term)

    J = J / m

    # ��ʧ����ƫ����
    J += (float(learning_rate) / (2 * m)) * \
        (np.sum(np.power(theta1[:, 1:], 2)) +
         np.sum(np.power(theta2[:, 1:], 2)))

    # ���򴫲�
    for t in range(m):
        # ��ÿ��ͼƬ���������������Ӧ�Ħģ��Ծ���ʽ���棩
        a1t = a1[t, :]  # (1, 401)
        z2t = z2[t, :]  # (1, 25)
        a2t = a2[t, :]  # (1, 26)
        ht = h[t, :]  # (1, 10)
        yt = y[t, :]  # (1, 10)

        d3t = ht - yt  # (1, 10)

        z2t = np.insert(z2t, 0, values=np.ones(1))  # (1, 26)
        d2t = np.multiply((theta2.T * d3t.T).T,
                          sigmoid_gradient(z2t))  # (1, 26)

        delta1 = delta1 + (d2t[:, 1:]).T * a1t
        delta2 = delta2 + d3t.T * a2t

    delta1 = delta1 / m
    delta2 = delta2 / m

    # ���򴫲��Ħ�����ƫ����
    delta1[:, 1:] = delta1[:, 1:] + (theta1[:, 1:] * learning_rate) / m
    delta2[:, 1:] = delta2[:, 1:] + (theta2[:, 1:] * learning_rate) / m

    # �����ݶ�
    grad = np.concatenate((np.ravel(delta1), np.ravel(delta2)))

    return J, grad
