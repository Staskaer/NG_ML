# ���ݼ���.m�ļ�������500��20*20���ص���д����ͼ��
# ����0�Ķ�Ӧֵ��10
# ��Ҫ��scipy��ȡ�ļ�

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from scipy.io import loadmat


def get_data():
    data = loadmat(r"ex3\ex3data1.mat")

    # ���չʾ100������
    sample_idx = np.random.choice(np.arange(data['X'].shape[0]), 100)
    sample_img = data['X'][sample_idx, :]

    # fig, ax_array = plt.subplots(
    #     nrows=10, ncols=10, sharey=True, sharex=True, figsize=(12, 12))

    # for r in range(10):
    #     for c in range(10):
    #         ax_array[r, c].matshow(np.array(
    #             sample_img[10 * r + c].reshape((20, 20))).T, cmap=matplotlib.cm.binary)
    #         plt.xticks(np.array([]))
    #         plt.yticks(np.array([]))
    # plt.show()

    return data
