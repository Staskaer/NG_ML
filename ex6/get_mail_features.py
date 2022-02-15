# 获取垃圾邮件的特征
from scipy.io import loadmat


def get_spam():
    spam_train = loadmat('ex6\spamTrain.mat')
    spam_test = loadmat('ex6\spamTest.mat')
    X = spam_train['X']
    Xtest = spam_test['Xtest']
    y = spam_train['y'].ravel()
    ytest = spam_test['ytest'].ravel()

    return (X, y, Xtest, ytest)
