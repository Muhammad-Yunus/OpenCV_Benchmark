import os
import cv2
import numpy as np 
from gst_cam import camera

# host mem not implemented, manually pin memory
class PinnedMem(object):
    def __init__(self, h, w, dtype=cv2.CV_8UC3):
        self.mem = cv2.cuda_HostMem(h, w, dtype, cv2.cuda.HostMem_PAGE_LOCKED)
        self.array = self.mem.createMatHeader()
        self.pinned = True

    def __del__(self):
        cv2.cuda.unregisterPageLocked(self.array)
        self.pinned = False
        
    def __repr__(self):
        return f'pinned = {self.pinned}'


class Pipeline: 
    def __init__(self, w=1280, h=720, multiplier=2, lr=0.05):
        self.cap            = cv2.VideoCapture(camera(0, w, h, fs=120), cv2.CAP_GSTREAMER)
        self.w, self.h      = w, h
        self.multiplier     = multiplier
        self.mog            = cv2.cuda.createBackgroundSubtractorMOG2()
        self.lr             = lr
        self.imgs_in        = [PinnedMem(self.w, self.h) for __ in range(2)]
        self.img            = cv2.cuda_GpuMat()
        self.img.create((self.w, self.h), cv2.CV_8UC3)
        self.gray           = cv2.cuda_GpuMat()
        self.gray.create((self.w, self.h), cv2.CV_8UC1)
        self.resizeUp       = cv2.cuda_GpuMat()
        self.resizeUp.create((self.w*self.multiplier, self.h*self.multiplier), cv2.CV_8UC1)
        self.mog_img        = cv2.cuda_GpuMat()
        self.mog_img.create((self.w*self.multiplier, self.h*self.multiplier), cv2.CV_8UC1)
        self.resizeDown     = cv2.cuda_GpuMat()
        self.resizeDown.create((self.w, self.h), cv2.CV_8UC1)

        self.stream         = cv2.cuda_Stream()
        self.is_next_frame  = False
        self.memory_index   = 0

    def apply(self):
        ret, __             = self.cap.read(self.getFrame())
        if not ret : 
            raise Exception("Invalid image frame!")

        if (self.is_next_frame) : 
           self.stream.waitForCompletion() # wait after we have read the next frame
        else : 
           self.is_next_frame = True

        self.img.upload(self.imgs_in[self.memory_index].array, stream=self.stream)
        cv2.cuda.cvtColor(self.img, cv2.COLOR_BGR2GRAY, self.gray, stream=self.stream)
        cv2.cuda.resize(self.gray, (self.w*self.multiplier, self.h*self.multiplier), self.resizeUp, stream=self.stream)
        self.mog.apply(self.resizeUp, fgmask=self.mog_img, learningRate = self.lr, stream=self.stream)
        cv2.cuda.resize(self.mog_img, (self.w, self.h), self.resizeDown, stream=self.stream)
        self.stream.queryIfComplete()
            
    def getFrame(self):
        self.memory_index = (self.memory_index + 1) % len(self.imgs_in)
        return self.imgs_in[self.memory_index].array

    def sync(self):
        self.stream.waitForCompletion()

    def close(self): 
        self.cap.release()

pipeline = Pipeline()

times = []
for i in range (1000):
    e1 = cv2.getTickCount()
    pipeline.apply()
    e2 = cv2.getTickCount()
    times.append((e2 - e1)/ cv2.getTickFrequency())

pipeline.sync()
time_avg = np.array(times).mean()

name = os.path.splitext(os.path.basename(__file__))[0]
print("%s : Execution Time : %.4f (FPS %.2f)" % (name, time_avg, (1/time_avg)))

pipeline.close()