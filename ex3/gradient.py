# ʵ���ݶȼ���

# ���Ƕ���������ǰ��������ص㿴���ǲ�����Ȼ����ͬ����Ŀ�Ĳ���
# ��֮ǰ��gradient����û��𣬵��Ǵ˴�ʹ�þ���������򻯱�ʾ��
# ��ʽ���Ӽ��
import numpy as np
from sigmoid import sigmoid


def gradient(theta, x, y, learning_rate):
    theta = np.matrix(theta)
    x = np.matrix(x)
    y = np.matrix(y)

    #parameters = int(theta.ravel().shape[1])
    error = sigmoid(x*theta.T) - y

    # grad = 1/m*x.T*(h(x)-y) + lambda/m*theta
    grad = ((x.T*error)/len(x)).T + ((learning_rate/(len(x)))*theta)

    # ƫ�����ҪԼ��
    grad[0, 0] = np.sum(np.multiply(error, x[:, 0]))/len(x)

    return np.array(grad).ravel()
