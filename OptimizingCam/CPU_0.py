import os
import cv2
import numpy as np 
from gst_cam import camera

#h, w                = 480, 320 
h, w                = 640, 480
multiplier          = 2
cap                 = cv2.VideoCapture(camera(0, w, h, fs=120), cv2.CAP_GSTREAMER)

times = []
for i in range (1000):
    
    e1                = cv2.getTickCount()
    ret, img          = cap.read()
    if not ret :
        raise Exception("Invalid image frame!")
    # gray            = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # resizeUp        = cv2.resize(gray, (w*multiplier, h*multiplier))
    # threshold, __   = cv2.threshold(resizeUp, 127, 255, cv2.THRESH_BINARY)
    # resizeDown      = cv2.resize(threshold, (w, h))
    e2                = cv2.getTickCount()
    times.append((e2 - e1)/ cv2.getTickFrequency())

time_avg = np.array(times).mean()

name = os.path.splitext(os.path.basename(__file__))[0]
print("%s : Execution Time : %.4f (FPS %.2f)" % (name, time_avg, (1/time_avg)))

cap.release()