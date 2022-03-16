# 读取多个参数时的数据
import pandas as pd
from featureNormalize import feature_normalize


def plot_data_multi():
    path = r"ex1\ex1data2.txt"
    data = pd.read_csv(path, header=None, names=['size', 'bedroom', 'price'])

    return data
