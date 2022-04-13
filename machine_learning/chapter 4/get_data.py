# 获取决策树的数据
import pandas as pd


def get_data():
    data = pd.read_csv(r"machine_learning\chapter 4\data.txt",
                       header=None, names=['color', 'root', 'sound', 'texture', 'umbilical', 'feel', 'y'], encoding='gbk')
    # x = data.iloc[:, :-1]
    # y = data.iloc[:, -1]
    # train = data.loc[{0, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15}]
    test = data.loc[{1, 4, 13, 16}]
    train = data
    return (train, test)
