# ������Ϣ�أ���Ϣ���桢�����ʡ�����ָ���ĺ���

import numpy as np


def compute_ent(D, a=0):
    # ������Ϣ��
    # p������ռ��
    # ���ü����ÿһ��������ķ�ʽ������ent
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
    # ������Ϣ����
    count_all = len(D)
    a_value_count = len(set(D[a].to_list()))  # ����a��ȡֵ��Χ
    a_type = list(set(D[a].to_list()))

    gain = 0
    iv = 0
    for value in range(a_value_count):
        # ����p*ent(Dv)
        p = len(D[D[a].isin([a_type[value]])])/count_all
        gain += p*compute_ent(D[D[a].isin([a_type[value]])])
        if Gain_flag is True:
            # ��ʱʹ����������Ϊ����׼��
            iv += p*np.log2(p)

    gain = compute_ent(D)-gain
    if Gain_flag is True:
        gain = gain/iv

    return -gain


def compute_gini(D, a):
    # �������ָ��

    def compute_gini_single(D):
        # ���ڼ���ÿ�������µĻ���ָ��
        total = len(D)
        positive = len(D[D['y'].isin([1])])/total
        negative = len(D[D['y'].isin([0])])/total
        gini = 1-positive*positive-negative*negative
        return gini

    count_all = len(D)
    a_value_count = len(set(D[a].to_list()))  # ����a��ȡֵ��Χ
    a_type = list(set(D[a].to_list()))

    gini = 0
    for value in range(a_value_count):
        # ����p*ent(Dv)
        p = len(D[D[a].isin([a_type[value]])])/count_all
        gini += p*compute_gini_single(D[D[a].isin([a_type[value]])])
    return gini
