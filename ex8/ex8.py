# 异常检测与推荐系统
from get_data import get_data_anomaly
from compute_gaussian_params import estimate_gaussian
from scipy import stats
import matplotlib.pyplot as plt
import numpy as np

from select_threshold import select_threshold


def anomaly_detection():
    data = get_data_anomaly()
    x = data['X']
    x_val = data['Xval']
    y_val = data['yval']

    mu, sigma = estimate_gaussian(x, False)
    # dist = stats.norm(mu[0], sigma[0])

    p = np.zeros((x.shape[0], x.shape[1]))
    p[:, 0] = stats.norm(mu[0], sigma[0]).pdf(x[:, 0])
    p[:, 1] = stats.norm(mu[1], sigma[1]).pdf(x[:, 1])

    p_val = np.zeros((x.shape[0], x.shape[1]))
    p_val[:, 0] = stats.norm(mu[0], sigma[0]).pdf(x_val[:, 0])
    p_val[:, 1] = stats.norm(mu[1], sigma[1]).pdf(x_val[:, 1])
    epsilon, f1 = select_threshold(p_val, y_val)
    out_liers = np.where(p < epsilon)

    # 绘图
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.scatter(x[:, 0], x[:, 1])
    ax.scatter(x[out_liers[0], 0], x[out_liers[0], 1],
               s=50, color='r', marker='o')
    plt.show()


if __name__ == "__main__":
    anomaly_detection()
