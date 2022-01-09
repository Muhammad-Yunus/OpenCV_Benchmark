import cv2
import numpy as np 
import os

img = cv2.imread("../lenna.png")
h, w, c = img.shape
multiplier = 2

times = []
for i in range (1000):
    e1              = cv2.getTickCount()
    gray            = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    resizeUp        = cv2.resize(gray, (w*multiplier, h*multiplier))
    threshold, __   = cv2.threshold(resizeUp, 127, 255, cv2.THRESH_BINARY)
    resizeDown      = cv2.resize(threshold, (w, h))
    e2              = cv2.getTickCount()
    times.append((e2 - e1)/ cv2.getTickFrequency())

time_avg = np.array(times).mean()

name = os.path.splitext(os.path.basename(__file__))[0]
print("%s : Execution Time : %.4f (FPS %.2f)" % (name, time_avg, (1/time_avg)))
