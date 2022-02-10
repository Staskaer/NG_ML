# 尝试实现绘图识别
from cv_theta import PredictTheta
from cv_mouse import Draw
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


predict_number()
