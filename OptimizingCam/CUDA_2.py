import os
import cv2
import numpy as np 
from gst_cam import camera

class Pipeline: 
    def __init__(self, w=1280, h=720, multiplier=2, lr=0.05):
        self.cap            = cv2.VideoCapture(camera(0, w, h, fs=120), cv2.CAP_GSTREAMER)
        self.w, self.h      = w, h
        self.multiplier     = multiplier
        self.mog            = cv2.cuda.createBackgroundSubtractorMOG2()
        self.lr             = lr
        self.img_in         = np.empty((self.h, self.w, 3), np.uint8)
        self.img            = cv2.cuda_GpuMat()
        self.img.create((self.h, self.w), cv2.CV_8UC3)
        self.gray           = cv2.cuda_GpuMat()
        self.gray.create((self.h, self.w), cv2.CV_8UC1)
        self.resizeUp       = cv2.cuda_GpuMat()
        self.resizeUp.create((self.h*self.multiplier, self.w*self.multiplier), cv2.CV_8UC1)
        self.mog_img        = cv2.cuda_GpuMat()
        self.mog_img.create((self.h*self.multiplier, self.w*self.multiplier), cv2.CV_8UC1)
        self.resizeDown     = cv2.cuda_GpuMat()
        self.resizeDown.create((self.h, self.w), cv2.CV_8UC1)

        self.stream = cv2.cuda_Stream()

    def apply(self):
        ret, __             = self.cap.read(self.img_in)
        if not ret : 
            raise Exception("Invalid image frame!")
        self.img.upload(self.img_in, stream=self.stream)
        cv2.cuda.cvtColor(self.img, cv2.COLOR_BGR2GRAY, self.gray, stream=self.stream)
        cv2.cuda.resize(self.gray, (self.w*self.multiplier, self.h*self.multiplier), self.resizeUp, stream=self.stream)
        self.mog.apply(self.resizeUp, fgmask=self.mog_img, learningRate = self.lr, stream=self.stream)
        cv2.cuda.resize(self.mog_img, (self.w, self.h), self.resizeDown, stream=self.stream)
            
    def close(self): 
        self.cap.release()

pipeline = Pipeline()

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