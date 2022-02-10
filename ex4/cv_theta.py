# 将神经网络预测和opencv相结合实现涂抹预测

import cv2
import numpy as np
# from getData import get_data
from sigmoid import sigmoid


class PredictTheta():
    def __init__(self) -> None:
        self.theta1 = np.load(r"ex4\theta1.npy")
        self.theta2 = np.load(r"ex4\theta2.npy")

    def predict(self, img):
        # 输入400单位的矩阵，输出评估数字
        img = np.matrix(img)
        a1 = np.insert(img, 0, 1)
        z2 = a1*self.theta1.T
        a2 = sigmoid(z2)
        a2 = np.insert(a2, 0, 1)
        z3 = a2*self.theta2.T
        a3 = sigmoid(z3)
        y_pred2 = np.argmax(a3, axis=1)+1
        return y_pred2 if y_pred2 != 10 else 0
