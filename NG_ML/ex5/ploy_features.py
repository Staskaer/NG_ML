# ���ļ��Ƕ���ʽ�ع�Ĳ���
from numpy.lib.npyio import load
from cost_reg import cost, cost_reg
from get_data import get_data
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from linear import linear_regression


# ����������������Ϊ������������׼��ѵ�����ݵ�


def ploy_features(x, power, as_ndarray=False):
    # ������������Ӳ�ͬ��������
    # ���ո����Ĵ���������Ҫ������
    data = {'f{}'.format(i): np.power(x, i) for i in range(1, power + 1)}
    df = pd.DataFrame(data)

    return df.values if as_ndarray else df


def normalize_features(df):
    # ��һ������
    return df.apply(lambda column: (column - column.mean())/column.std())


def prepare_ploy_data(*args, power):
    # Ϊѵ��ģ��׼������

    def prepare(x):
        df = ploy_features(x, power)
        ndarr = normalize_features(df).values
        return np.insert(ndarr, 0, np.ones(ndarr.shape[0]), axis=1)

    return [prepare(x)for x in args]


# ���µĺ�������������ѵ����


def plot_learning_curve(x, x_init, y, x_val, y_val, l=0, power=8):
    train_cost, cv_cost = [], []
    m = x.shape[0]

    for i in range(1, 1+m):
        res = linear_regression(x[:i, :], y[:i], l)
        tc = cost_reg(res.x, x[:i, :], y[:i], l)
        cv = cost_reg(res.x, x_val, y_val, l)

        train_cost.append(tc)
        cv_cost.append(cv)

    # ������չʾ�����õ�
    fig, ax = plt.subplots(2,  1, figsize=(12, 12))
    ax[0].plot(np.arange(1, m + 1), train_cost, label='training cost')
    ax[0].plot(np.arange(1, m + 1), cv_cost, label='cv cost')
    ax[0].legend()
    fitx = np.linspace(-50, 50, 100)
    fitxtmp = prepare_ploy_data(fitx, power=power)
    fity = np.dot(prepare_ploy_data(fitx, power=power)[
                  0], linear_regression(x, y, l).x.T)

    ax[1].plot(fitx, fity, c='r', label='fitcurve')
    ax[1].scatter(x_init, y, c='b', label='initial_Xy')

    ax[1].set_xlabel('water_level')
    ax[1].set_ylabel('flow')


def find_lambda(x, y, x_val, y_val):
    l_candidate = [0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10]
    training_cost, cv_cost = [], []

    for l in l_candidate:
        res = linear_regression(x, y, l)

        tc = cost_reg(res.x, x, y, l)
        cv = cost_reg(res.x, x_val, y_val, l)

        training_cost.append(tc)
        cv_cost.append(cv)

    # ����
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot(l_candidate, training_cost, label='training')
    ax.plot(l_candidate, cv_cost, label='cross validation')
    plt.legend()
    plt.xlabel('lambda')
    plt.ylabel('cost')
    plt.show()


# ���º����Ƕ�����ѵ����ͼ����Ƶ���ں���


def ploy_features_main(reg=0, power=8, type=0):
    # ��ں������Կ��ƴ�����Լ��ϵ��
    x, y, x_val, y_val, x_test, y_test = get_data()

    x_ploy, x_val_ploy, x_test_ploy = prepare_ploy_data(
        x, x_val, x_test, power=power)

    if type is 0:
        plot_learning_curve(x_ploy, x, y, x_val_ploy,
                            y_val, l=reg, power=power)
    else:
        find_lambda(x_ploy, y, x_val_ploy, y_val)
    plt.show()
