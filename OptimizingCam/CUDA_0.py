import os
import cv2
import numpy as np 
from gst_cam import camera

#h, w                    = 480, 320 
h, w                    = 640, 480
cap                     = cv2.VideoCapture(camera(0, w, h, fs=120), cv2.CAP_GSTREAMER)
multiplier              = 2
imgGPU                  = cv2.cuda_GpuMat()

times = []
for i in range (1000):
    e1                  = cv2.getTickCount()
    ret, img            = cap.read()
    if not ret : 
        raise Exception("Invalid image frame!")
    imgGPU.upload(img)
    grayGPU             = cv2.cuda.cvtColor(imgGPU, cv2.COLOR_BGR2GRAY)
    resizeUpGPU         = cv2.cuda.resize(grayGPU, (w*multiplier, h*multiplier))
    __, thresholdGPU    = cv2.cuda.threshold(resizeUpGPU, 127, 255, cv2.THRESH_BINARY)
    resizeDownGPU       = cv2.cuda.resize(resizeUpGPU, (w, h))
    e2                  = cv2.getTickCount()
    times.append((e2 - e1)/ cv2.getTickFrequency())

time_avg = np.array(times).mean()

name = os.path.splitext(os.path.basename(__file__))[0]
print("%s : Execution Time : %.4f (FPS %.2f)" % (name, time_avg, (1/time_avg)))