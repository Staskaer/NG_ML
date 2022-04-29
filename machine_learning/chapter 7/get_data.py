# 获取贝叶斯分类器的数据
import pandas as pd
import numpy as np


def get_data():
    data = pd.read_csv(r"machine_learning\chapter 4\data.txt", header=None, names=[
                       'color', 'root', 'sound', 'texture', 'umbilical', 'feel', 'y'])
    test = data.loc[{1, 4, 13, 16}]
    data = data.drop({1, 4, 13, 16})
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]
    X_test = test.iloc[:, :-1]
    y_test = test.iloc[:, -1]
    return (X, y, X_test, y_test)
