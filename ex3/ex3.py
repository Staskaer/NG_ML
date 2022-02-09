# 实现手写数字识别

import numpy as np
from getData import get_data
from one_vs_all import one_vs_all
from predict_all import predict_all
from sklearn.metrics import classification_report

if __name__ == "__main__":
    data = get_data()
    params = data['X'].shape[1]
    all_theta = np.zeros((10, params+1))
    # rows = data['X'].shape[0]
    # x = np.insert(data['X'], 0, values=np.ones(rows), axis=1)

    # theta = np.zeros(params+1)

    # y_0 = np.array([1 if label == 0 else 0 for label in data['y']])
    # y_0 = np.reshape(y_0, (rows, 1))

    # 怪
    all_theta = one_vs_all(data['X'], data['y'], 10, 1)

    y_pred = predict_all(data['X'], all_theta=all_theta)
    print(classification_report(data['y'], y_pred))
