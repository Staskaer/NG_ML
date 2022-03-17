# 计算信息熵，信息增益、增益率、基尼指数的函数

import numpy as np


def compute_ent(D, a=0):
    # 计算信息熵
    # p是类别的占比
    # 采用计算出每一类别数量的方式来计算ent
    total = len(D)
    positive = len(D[D['y'].isin([1])])
    negative = len(D[D['y'].isin([0])])
    if positive == 0:
        positive = total
    if negative == 0:
        negative = total

    ent = positive/total*np.log2(positive/total) + \
        negative/total*np.log2(negative/total)
    return -ent


def compute_gain(D, a, Gain_flag=False):
    # 计算信息增益
    count_all = len(D)
    a_value_count = len(set(D[a].to_list()))  # 属性a的取值范围
    a_type = list(set(D[a].to_list()))

    gain = 0
    iv = 0
    for value in range(a_value_count):
        # 计算p*ent(Dv)
        p = len(D[D[a].isin([a_type[value]])])/count_all
        gain += p*compute_ent(D[D[a].isin([a_type[value]])])
        if Gain_flag is True:
            # 此时使用增益率作为计算准则
            iv += p*np.log2(p)

    gain = compute_ent(D)-gain
    if Gain_flag is True:
        gain = gain/iv

    return -gain


def compute_gini(D, a):
    # 计算基尼指数

    def compute_gini_single(D):
        # 用于计算每个划分下的基尼指数
        total = len(D)
        positive = len(D[D['y'].isin([1])])/total
        negative = len(D[D['y'].isin([0])])/total
        gini = 1-positive*positive-negative*negative
        return gini

    count_all = len(D)
    a_value_count = len(set(D[a].to_list()))  # 属性a的取值范围
    a_type = list(set(D[a].to_list()))

    gini = 0
    for value in range(a_value_count):
        # 计算p*ent(Dv)
        p = len(D[D[a].isin([a_type[value]])])/count_all
        gini += p*compute_gini_single(D[D[a].isin([a_type[value]])])
    return gini
