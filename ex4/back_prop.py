# 反向传播

import numpy as np
from forward_function import forward
from sigmoid import sigmoid_gradient
from cost_reg import cost_reg


def back_prop(params, input_size, hidden_size, num_labels, x, y, learning_rate):
    m = x.shape[0]
    x = np.matrix(x)
    y = np.matrix(y)

    # 先把theta从params中提取出来
    theta1 = np.matrix(np.reshape(
        params[:hidden_size * (input_size + 1)], (hidden_size, (input_size + 1))))
    theta2 = np.matrix(np.reshape(
        params[hidden_size * (input_size + 1):], (num_labels, (hidden_size + 1))))

    # 前向传播
    a1, z2, a2, z3, h = forward(x, theta1, theta2)

    # 预备参数
    j = 0
    delta1 = np.zeros(theta1.shape)
    delta2 = np.zeros(theta2.shape)

    # 计算损失
    j = cost_reg(theta1, theta2, input_size, hidden_size,
                 num_labels, x, y, learning_rate)

    # for i in range(m):
    #     first_term = np.multiply(-y[i, :], np.log(h[i, :]))
    #     second_term = np.multiply((1-y[i, :]), np.log(1-h[i, :]))
    #     j += np.sum(first_term-second_term)

    # j = j/m

    # j += (float(learning_rate) / (2 * m)) * \
    #     (np.sum(np.power(theta1[:, 1:], 2)) +
    #      np.sum(np.power(theta2[:, 1:], 2)))

    # 执行反向传播
    for t in range(m):
        a1t = a1[t, :]  # (1,401)
        z2t = z2[t, :]  # (1,25)
        a2t = a2[t, :]  # (1,26)
        ht = h[t, :]  # (1,10)
        yt = y[t, :]  # (1,10)

        # 这些全是公式
        d3t = ht-yt  # (1,10)
        z2t = np.insert(z2t, 0, values=np.ones(1))  # (1,26)
        d2t = np.multiply((theta2.T * d3t.T).T,
                          sigmoid_gradient(z2t))  # (1, 26)

        delta1 += (d2t[:, 1:]).T*a1t
        delta2 += d3t.T*a2t

    delta1 = delta1 / m
    delta2 = delta2 / m

    return j, delta1, delta2
