# ����ʵ�ֻ�ͼʶ��
from cv_theta import PredictTheta
from cv_mouse import Draw
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


predict_number()
