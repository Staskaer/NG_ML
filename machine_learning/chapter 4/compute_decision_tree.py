# ʵ�ּ���������ļ��㺯��

from utils import compute_ent, compute_gain, compute_gini
from choose import choose_best_attr, judge_equal, compute_major_label, compute_equal_attr


def compute_decision_tree(train, test, names, del_name=[], func=compute_gain):
    # �����������������չ�ɼ�֦ģʽ
    tree = {}
    names = [name for name in names if name not in del_name]

    if judge_equal(train) is 1:
        # ȫ����ǩһ����û��Ҫ�ٷ�
        return compute_major_label(train)

    if len(train) == 1 or compute_equal_attr(train, del_name):
        # ֻʣ��һ����ǩ������ȫ��һ��
        return compute_major_label(train)

    attr = choose_best_attr(train, names, func=func)
    # ������õ�����
    tree[attr] = {}

    labels = list(set(train[attr]))
    # ��ȡ��ǰ���ԵĿ���ֵ

    for l in labels:
        del_name.append(attr)
        subtree = compute_decision_tree(
            train[train[attr].isin([l])], test, names, del_name, func)
        tree[attr][l] = subtree
    return tree
