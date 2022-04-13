# 决策树

from get_data import get_data
from utils import compute_gini, compute_ent, compute_gain
from compute_decision_tree import compute_decision_tree
from show import createPlot


if __name__ == "__main__":
    train, test = get_data()
    names = ['color', 'root', 'sound', 'texture', 'umbilical', 'feel']

    tree = compute_decision_tree(train, test, names)
    print(tree)
    createPlot(tree)
