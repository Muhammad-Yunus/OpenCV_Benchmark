import cv2
import numpy as np 
import os

class Pipeline: 
    def __init__(self, img_in):
        self.multiplier             = 2
        self.h, self.w, self.c      = img_in.shape
        self.img                    = cv2.cuda_GpuMat()
        self.img.create((self.w, self.h), cv2.CV_8UC3)
        self.img.upload(img_in)

        self.gray                   = cv2.cuda_GpuMat()
        self.gray.create((self.w, self.h), cv2.CV_8UC1)
        self.resizeUp               = cv2.cuda_GpuMat()
        self.resizeUp.create((self.w*self.multiplier, self.h*self.multiplier), cv2.CV_8UC1)
        self.mog_img                = cv2.cuda_GpuMat()
        self.mog_img.create((self.w*self.multiplier, self.h*self.multiplier), cv2.CV_8UC1)
        self.resizeDown             = cv2.cuda_GpuMat()
        self.resizeDown.create((self.w, self.h), cv2.CV_8UC1)

    def apply(self):
        cv2.cuda.cvtColor(self.img, cv2.COLOR_BGR2GRAY, self.gray)
        cv2.cuda.resize(self.gray, (self.w*self.multiplier, self.h*self.multiplier), self.resizeUp)
        cv2.cuda.threshold(self.resizeUp, 127, 255, cv2.THRESH_BINARY, self.threshold)
        cv2.cuda.resize(self.threshold, (self.w, self.h), self.resizeDown)
            

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