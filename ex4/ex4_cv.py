# 尝试实现绘图识别
from cv_theta import PredictTheta
from cv_mouse import Draw
from scipy.io import loadmat
import numpy as np
import cv2


def predict_number():
    # 这段代码是用opencv实现的绘图窗口
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


def predict_raw():
    # 测试原始数据的判断
    # 发现之前由于重大失误，把数据的行列搞反了
    # 已修正，但是效果依然不是很好
    # 原因有下：
    # cv2的resize导致部分像素丢失
    # 绘图的笔迹像素与原来的像素差距较大

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
