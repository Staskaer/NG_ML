# һ�Զ������
# �����߼��ع�ֻ����������֮����з���
# ��Ҫ�����������������ѵ��
import numpy as np
from cost import cost
from gradient import gradient
from scipy.optimize import minimize


def one_vs_all(x, y, num_labels, learning_rate):
    rows = x.shape[0]
    params = x.shape[1]

    # k�����k*(n-1)�ľ���
    all_theta = np.zeros((num_labels, params + 1))

    # �����һ��
    x = np.insert(x, 0, values=np.ones(rows), axis=1)

    for i in range(1, num_labels+1):
        theta = np.zeros(params+1)
        y_i = np.array([1 if label == i else 0 for label in y])
        # ��y_i����ǵ�ǰ������������һ��

        y_i = np.reshape(y_i, (rows, 1))

        fmin = minimize(fun=cost, x0=theta, args=(
            x, y_i, learning_rate), method='TNC', jac=gradient)
        all_theta[i-1, :] = fmin.x

    return all_theta
