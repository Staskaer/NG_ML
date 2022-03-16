# ����ʵ�ֻ�ͼʶ��
from cv_theta import PredictTheta
from cv_mouse import Draw
from scipy.io import loadmat
import numpy as np
import cv2


def predict_number():
    # ��δ�������opencvʵ�ֵĻ�ͼ����
    # ��2ʶ�𣬰�1���
    drawer = Draw()
    predict_theta = PredictTheta()
    while 1:
        key = cv2.waitKey(0)
        if key == ord('1'):
            drawer.clear()
        if key == ord('2'):
            num = predict_theta.predict(drawer.data)
            print(num)
            drawer.put_text(num)


def predict_raw():
    # ����ԭʼ���ݵ��ж�
    # ����֮ǰ�����ش�ʧ�󣬰����ݵ����и㷴��
    # ������������Ч����Ȼ���Ǻܺ�
    # ԭ�����£�
    # cv2��resize���²������ض�ʧ
    # ��ͼ�ıʼ�������ԭ�������ز��ϴ�

    data = loadmat(r"ex4\ex4data1.mat")
    img2 = data['X']
    # img2 = (img2-np.min(img2))/(np.max(img2)-np.min(img2))
    # img2 = np.float64(img2 > 0.5)

    a = PredictTheta()
    for i in range(img2.shape[0]):
        img = img2[i]
        img = np.reshape(img, (20, 20))
        img = img.T
        num = a.predict(img)

        img = cv2.resize(img, (400, 400))
        cv2.putText(img, str(num), (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, 1, 2)

        cv2.imshow("img", img)
        cv2.waitKey(-1)


predict_raw()
