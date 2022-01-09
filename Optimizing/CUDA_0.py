import cv2
import numpy as np 
import os

img = cv2.imread("../lenna.png")
imgGPU = cv2.cuda_GpuMat()
imgGPU.upload(img)
h, w, c = img.shape
multiplier = 2

times = []
for i in range (1000):
    e1                  = cv2.getTickCount()
    grayGPU             = cv2.cuda.cvtColor(imgGPU, cv2.COLOR_BGR2GRAY)
    resizeUpGPU         = cv2.cuda.resize(grayGPU, (w*multiplier, h*multiplier))
    __, thresholdGPU    = cv2.cuda.threshold(resizeUpGPU, 127, 255, cv2.THRESH_BINARY)
    resizeDownGPU       = cv2.cuda.resize(resizeUpGPU, (w, h))
    e2                  = cv2.getTickCount()
    times.append((e2 - e1)/ cv2.getTickFrequency())

time_avg = np.array(times).mean()

name = os.path.splitext(os.path.basename(__file__))[0]
print("%s : Execution Time : %.4f (FPS %.2f)" % (name, time_avg, (1/time_avg)))