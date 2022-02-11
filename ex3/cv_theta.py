# 将神经网络预测和opencv相结合实现涂抹预测

import cv2
import numpy as np
from get_weight import get_weight
from getData import get_data
from sigmoid import sigmoid


class PredictTheta():
    def __init__(self) -> None:
        self.theta1, self.theta2 = get_weight()

    def predict(self, img):
        # 输入400单位的矩阵，输出评估数字
        img = np.matrix(img)
        img = img.T

        a1 = np.insert(img, 0, 1)
        z2 = a1*self.theta1.T
        a2 = sigmoid(z2)
        a2 = np.insert(a2, 0, 1)
        z3 = a2*self.theta2.T
        a3 = sigmoid(z3)
        y_pred2 = np.argmax(a3, axis=1)+1
        return y_pred2 if y_pred2 != 10 else 0


# data = get_data()
# img = data['X'][0]

# img = np.reshape(img, (20, 20))
# print(img)
# img = cv2.resize(img, (400, 400))

# cv2.imshow("img", img)
# cv2.waitKey(-1)
