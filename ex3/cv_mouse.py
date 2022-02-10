# 定义绘制窗口的类

import cv2
import numpy as np


class Draw():
    def __init__(self) -> None:
        self.img = np.zeros((400, 400), dtype=np.float64)
        self.window = "number"
        self.flag = False
        self.x = 0
        self.y = 0
        self.data = np.zeros((1, 400), dtype=np.float64)

        cv2.namedWindow(self.window)
        cv2.setMouseCallback(self.window, self.on_mouse)

    def on_mouse(self, event, x, y, flags, param):
        self.x, self.y = x, y
        if event == cv2.EVENT_LBUTTONDOWN:
            self.flag = True
        if event == cv2.EVENT_LBUTTONUP:
            self.flag = False
        self.draw()

    def clear(self):
        self.img = np.zeros((400, 400), dtype=np.float64)
        self.flag = False
        self.x = 0
        self.y = 0
        self.data = np.zeros((1, 400), dtype=np.float64)
        self.draw()

    def draw(self):
        if self.flag == True:
            self.img[self.y-5:self.y+5, self.x -
                     5:self.x+5] = np.ones(10)
        cv2.imshow(self.window, self.img)

        # 显示小图
        temp = cv2.resize(self.img, (20, 20))
        self.data = np.reshape(temp, (1, 400))
        cv2.imshow("temp", temp)

    def put_text(self, num):
        cv2.putText(self.img, str(num), (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, 1, 2)


# a = Draw()
# while 1:
#     key = cv2.waitKey(0)
#     if key == ord('1'):
#         a.clear()
#         a.put_text(1)
