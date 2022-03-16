# 估算高斯分布中的mu和sigma
import numpy as np
import matplotlib.pyplot as plt


def estimate_gaussian(x, draw=False):
    # 计算sigma与mu
    mu = x.mean(axis=0)
    sigma = x.var(axis=0)

    if draw is True:
        xplot = np.linspace(0, 25, 100)

        yplot = np.linspace(0, 25, 100)
        Xplot, Yplot = np.meshgrid(xplot, yplot)
        Z = np.exp((-0.5)*((Xplot-mu[0])**2 /
                           sigma[0]+(Yplot-mu[1])**2/sigma[1]))

        fig, ax = plt.subplots(figsize=(12, 8))
        contour = plt.contour(
            Xplot, Yplot, Z, [10**-11, 10**-7, 10**-5, 10**-3, 0.1], colors='k')
        ax.scatter(x[:, 0], x[:, 1])
        plt.show()

    return mu, sigma
