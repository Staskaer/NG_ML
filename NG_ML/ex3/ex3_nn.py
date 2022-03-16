# ���ø���������ʵ��ǰ��������

from get_weight import get_weight
from sklearn.metrics import classification_report
from getData import get_data
from sigmoid import sigmoid
from cv_theta import PredictTheta
from cv_mouse import Draw
import numpy as np
import cv2


def predict_number():
    # ��δ�������opencvʵ�ֵĻ�ͼ����
    # ���ǿ������ڴ洢���ݸ�ʽ����ƥ�䣬��������ʶ��
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


def main():
    data = get_data()
    theta1, theta2 = get_weight()

    x = np.matrix(
        np.insert(data['X'], 0, values=np.ones(data['X'].shape[0]), axis=1))
    y = np.matrix(data['y'])

    # ���¾���������ǰ�򴫲��Ĺ��̣�ע�����ƫ����
    a1 = x
    z2 = a1*theta1.T

    a2 = sigmoid(z2)
    a2 = np.insert(a2, 0, values=np.ones(a2.shape[0]), axis=1)
    z3 = a2*theta2.T

    a3 = sigmoid(z3)

    y_pred2 = np.argmax(a3, axis=1)+1
    print(classification_report(y, y_pred2))


if __name__ == "__main__":
    predict_number()
