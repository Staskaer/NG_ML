# 利用svm对邮件进行垃圾邮件判断
from sklearn import svm
import numpy as np
import pandas as pd
from get_mail_features import get_spam

if __name__ == "__main__":
    x, y, x_test, y_test = get_spam()
    svc = svm.SVC()
    # 训练数据
    svc.fit(x, y)
    print('Training accuracy = {0}%'.format(
        np.round(svc.score(x, y) * 100, 2)))
    print('Test accuracy = {0}%'.format(
        np.round(svc.score(x_test, y_test) * 100, 2)))

    # 可视化结果
    kw = np.eye(1899)
    spam_val = pd.DataFrame({"idx": range(1899)})
    spam_val['isspam'] = svc.decision_function(kw)
    print(spam_val['isspam'].describe())
