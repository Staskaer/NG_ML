# 计算合适的阈值
import numpy as np


def select_threshold(p_val, y_val):
    best_epsilon = 0
    best_f1 = 0
    f1 = 0

    step = (p_val.max()-p_val.min())/1000

    for epsilon in np.arange(p_val.min(), p_val.max(), step):
        preds = p_val < epsilon

        tp = np.sum(np.logical_and(preds == 1, y_val == 1)).astype(float)
        fp = np.sum(np.logical_and(preds == 1, y_val == 0)).astype(float)
        fn = np.sum(np.logical_and(preds == 0, y_val == 1)).astype(float)

        precision = tp/(tp+fp)
        recall = tp/(tp+fn)
        f1 = (2*precision*recall)/(precision+recall)

        if f1 > best_f1:
            best_f1 = f1
            best_epsilon = epsilon

    return best_epsilon, best_f1
