# ʵ�������ʼ��
import numpy as np


def random_init(hidden_size, input_size, num_label):
    # np.random.random(size)����size��С�����0-1������
    params = (np.random.random(size=hidden_size *
                               (input_size+1)+num_label*(hidden_size+1))-0.5)*0.24
