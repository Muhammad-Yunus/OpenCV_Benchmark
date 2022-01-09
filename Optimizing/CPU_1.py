import cv2
import numpy as np 
import os

class Pipeline: 
    def __init__(self, img):
        self.multiplier         = 2
        self.h, self.w, self.c  = img.shape
        self.img                = img
        self.gray               = np.empty(img.shape[:2], np.uint8)
        self.blur               = np.empty((self.h, self.w), np.uint8)
        self.resizeUp           = np.empty((self.h*self.multiplier, self.w*self.multiplier), np.uint8)
        self.threshold          = np.empty((self.h*self.multiplier, self.w*self.multiplier), np.uint8)
        self.resizeDown         = np.empty((self.h, self.w), np.uint8)

    def apply(self):
        cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY, self.gray)
        cv2.resize(self.gray, (self.w*self.multiplier, self.h*self.multiplier), self.resizeUp)
        self.threshold, __ = cv2.threshold(self.resizeUp, 127, 255, cv2.THRESH_BINARY)
        cv2.resize(self.threshold, (self.w, self.h), self.resizeDown)
        
img = cv2.imread("../lenna.png")
pipeline = Pipeline(img)

times = []
for i in range (1000):
    e1 = cv2.getTickCount()
    pipeline.apply()
    e2 = cv2.getTickCount()
    times.append((e2 - e1)/ cv2.getTickFrequency())

time_avg = np.array(times).mean()

name = os.path.splitext(os.path.basename(__file__))[0]
print("%s : Execution Time : %.4f (FPS %.2f)" % (name, time_avg, (1/time_avg)))