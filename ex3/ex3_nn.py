# 利用给定参数来实现前馈神经网络

from get_weight import get_weight
from sklearn.metrics import classification_report
from getData import get_data
from sigmoid import sigmoid
from cv_theta import PredictTheta
from cv_mouse import Draw
import numpy as np
import cv2


def predict_number():
    # 这段代码是用opencv实现的绘图窗口
    # 但是可能由于存储数据格式不破匹配，不能正常识别
    # 按2识别，按1清除
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

    # 以下就是神经网络前向传播的过程，注意添加偏置项
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
