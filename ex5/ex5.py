# ͨ�����������ѧϰѧϰ���ߡ���/Ƿ���֮�����ϵ

# ÿ���ļ��ڶ��д����򻯺Ͳ������򻯵�������ʽ
from gradient import gradient, gradient_reg
from cost_reg import cost, cost_reg
from ploy_features import ploy_features_main
from linear import linear
from get_data import get_data
import numpy as np

if __name__ == "__main__":
    # ��ȡ����
    x, y, x_val, y_val, x_test, y_test = get_data()
    x, x_val, x_test = [np.insert(x.reshape(x.shape[0], 1), 0, np.ones(
        x.shape[0]), axis=1) for x in (x, x_val, x_test)]

    x = np.matrix(x)
    y = np.matrix(y)
    theta = np.ones(x.shape[1])

    # ���Ƚ�������ģ�͵���ϲ���
    # ���Թ۲⵽����ѧϰ���߷ǳ��Ľӽ���˵������Ƿ���״̬

    #linear(x, y, theta, False, (x_val, y_val, x_test, y_test))

    # Ȼ��۲����ʽ���
    ploy_features_main(reg=10, power=10)
