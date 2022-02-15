# ��ȡ�����ļ�

import pandas as pd
import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt


def get_data_1(draw=False):
    # ��ȡdata1
    # ��Щ��������һ���������쳣�㣨����ƫ������λ�ã�
    raw_data = loadmat(r"ex6\ex6data1.mat")
    data = pd.DataFrame(raw_data.get("X"), columns=['X1', 'X2'])
    data['y'] = raw_data.get('y')
    print(data.head())

    if draw is True:
        positive = data[data['y'].isin([1])]
        negative = data[data['y'].isin([0])]

        fig, ax = plt.subplots(figsize=(12, 8))
        ax.scatter(positive['X1'], positive['X2'],
                   s=50, marker='x', label='positive')
        ax.scatter(negative['X1'], negative['X2'],
                   s=50, marker='o', label='negative')
        ax.legend()
        plt.show()

    return data


def get_data2(draw=False):
    # ��ȡdata2
    raw_data = loadmat(r"ex6\ex6data2.mat")
    data = pd.DataFrame(raw_data.get("X"), columns=['X1', 'X2'])
    data['y'] = raw_data.get('y')
    print(data.head())

    if draw is True:
        positive = data[data['y'].isin([1])]
        negative = data[data['y'].isin([0])]

        fig, ax = plt.subplots(figsize=(12, 8))
        ax.scatter(positive['X1'], positive['X2'],
                   s=50, marker='x', label='positive')
        ax.scatter(negative['X1'], negative['X2'],
                   s=50, marker='o', label='negative')
        ax.legend()
        plt.show()

    return data


def get_data3(draw=False):
    # ��ȡdata3
    raw_data = loadmat(r"ex6\ex6data3.mat")
    x = raw_data['X']
    x_val = raw_data['Xval']
    y = raw_data['y'].ravel()
    y_val = raw_data['yval'].ravel()
    data = pd.DataFrame(raw_data.get('X'), columns=['X1', 'X2'])
    data['y'] = raw_data.get("y")

    if draw is True:
        fig, ax = plt.subplots(figsize=(12, 8))
        positive = data[data['y'].isin([1])]
        negative = data[data['y'].isin([0])]
        ax.scatter(positive['X1'], positive['X2'],
                   s=50, marker='x', label='positive')
        ax.scatter(negative['X1'], negative['X2'],
                   s=50, marker='o', label='negative')
        ax.legend()
        plt.show()

    return (x, y, x_val, y_val, data)
