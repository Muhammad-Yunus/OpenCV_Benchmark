import os
import cv2
import numpy as np 
from gst_cam import camera

class Pipeline: 
    def __init__(self, w=480, h=320, multiplier=2):
        self.cap                = cv2.VideoCapture(camera(0, w, h, fs=120), cv2.CAP_GSTREAMER)
        self.w, self.h          = w, h
        self.multiplier         = multiplier
        self.img                = np.empty((self.h, self.w, 3), np.uint8)
        self.gray               = np.empty((self.h, self.w), np.uint8)
        self.blur               = np.empty((self.h, self.w), np.uint8)
        self.resizeUp           = np.empty((self.h*self.multiplier, self.w*self.multiplier), np.uint8)
        self.threshold          = np.empty((self.h*self.multiplier, self.w*self.multiplier), np.uint8)
        self.resizeDown         = np.empty((self.h, self.w), np.uint8)

    def apply(self):
        ret, __                 = self.cap.read(self.img)
        if not ret : 
            raise Exception("Invalid image frame!")
        cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY, self.gray)
        cv2.resize(self.gray, (self.w*self.multiplier, self.h*self.multiplier), self.resizeUp)
        self.threshold, __      = cv2.threshold(self.resizeUp, 127, 255, cv2.THRESH_BINARY)
        cv2.resize(self.threshold, (self.w, self.h), self.resizeDown)
        
    def close(self): 
        self.cap.release()

pipeline = Pipeline(w=640, h=480)

times = []
for i in range (1000):
    e1 = cv2.getTickCount()
    pipeline.apply()
    e2 = cv2.getTickCount()
    times.append((e2 - e1)/ cv2.getTickFrequency())

time_avg = np.array(times).mean()

name = os.path.splitext(os.path.basename(__file__))[0]
print("%s : Execution Time : %.4f (FPS %.2f)" % (name, time_avg, (1/time_avg)))

pipeline.close()