# 贝叶斯分类器
from AODE import AODE
from get_data import get_data
import numpy as np
from naive_bayes import naive_bayes


def naive_bayes_test():
    # 朴素贝叶斯的测试
    X, y, X_test, y_test = get_data()

    # 数据转换
    X = np.matrix(X)
    y = np.matrix(y)
    X_test = np.matrix(X_test)
    y_test = np.matrix(y_test)
    # print(X, y, X_test, y_test)

    # 调用朴素贝叶斯预测，和结果做对比
    result, p = naive_bayes(X, y, X_test)
    for i in range(y_test.shape[1]):
        print("当前正确值为{}，预测值为{}，概率为{}".format(
            y_test[0, i], result[i, 0], p[i, 0]))


def AODE_test():
    # AODE半朴素贝叶斯测试
    X, y, X_test, y_test = get_data()

    # 数据转换
    X = np.matrix(X)
    y = np.matrix(y)
    X_test = np.matrix(X_test)
    y_test = np.matrix(y_test)

    # 调用朴素贝叶斯预测，和结果做对比
    result, p = AODE(X, y, X_test)
    for i in range(y_test.shape[1]):
        print("当前正确值为{}，预测值为{}，概率为{}".format(
            y_test[0, i], result[i, 0], p[i, 0]))


if __name__ == "__main__":
    # naive_bayes_test()
    AODE_test()
    # aode应该是没有写错，但是样本太少了，不好预测捏
