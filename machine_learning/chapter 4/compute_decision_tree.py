# 实现计算决策树的计算函数

from utils import compute_ent, compute_gain, compute_gini
from choose import choose_best_attr, judge_equal, compute_major_label, compute_equal_attr


def compute_decision_tree(train, test, names, del_name=[], func=compute_gain):
    # 计算决策树，可以扩展成剪枝模式
    tree = {}
    names = [name for name in names if name not in del_name]

    if judge_equal(train) is 1:
        # 全部标签一样，没必要再分
        return compute_major_label(train)

    if len(train) == 1 or compute_equal_attr(train, del_name):
        # 只剩下一个标签或属性全部一致
        return compute_major_label(train)

    attr = choose_best_attr(train, names, func=func)
    # 计算最好的属性
    tree[attr] = {}

    labels = list(set(train[attr]))
    # 获取当前属性的可能值

    for l in labels:
        del_name.append(attr)
        subtree = compute_decision_tree(
            train[train[attr].isin([l])], test, names, del_name, func)
        tree[attr][l] = subtree
    return tree
