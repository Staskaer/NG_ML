# ��ex3���ƣ���ȡ����

from scipy.io import loadmat
# import cv2
import numpy as np
from sklearn.preprocessing import OneHotEncoder


def get_data():
    # ������ϴ��һ������
    data = loadmat(r"ex4\ex4data1.mat")
    img2 = data['X']
    y = data['y']

    # �˴������˹�һ����������˼������cost���ܻ��в���
    img2 = (img2-np.min(img2))/(np.max(img2)-np.min(img2))
    img2 = np.float64(img2 > 0.5)

    # �˴�������y��ά��
    encoder = OneHotEncoder(sparse=False)
    y_onehot = encoder.fit_transform(y)

    return (img2, y_onehot, y)


# ���Դ���
# data = load_mat()
# img2 = data['X']
# # img2 = np.array(img2)
# img2 = (img2-np.min(img2))/(np.max(img2)-np.min(img2))

# img2 = np.float64(img2 > 0.5)


# img = img2[0]
# print(img.shape)
# img = np.reshape(img, (20, 20))
# #img = (img-np.min(img))/(np.max(img)-np.min(img))
# dst = img.copy()
# # img = cv2.normalize(img, dst, 0, 1, cv2.NORM_MINMAX)
# img = cv2.resize(img, (400, 400))
# cv2.imshow("img", img)
# cv2.waitKey(-1)
