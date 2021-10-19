import cv2
import numpy as np 

class Pipeline: 
    def __init__(self, img):
        self.img = img
        self.h, self.w, self.c = img.shape
        self.multiplier = 2

        # define kernel
        #self.ocl_kernel = cv2.gapi.core.cpu.kernels() 
        self.ocl_kernel = cv2.gapi.core.ocl.kernels()
        
    def graph(self):
        # define graph
        img_GMat = cv2.GMat()
        maxval_sc = cv2.GScalar()
        gray_GMat = cv2.gapi.RGB2Gray(img_GMat)
        blur_GMat = cv2.gapi.medianBlur(gray_GMat, 3)
        resizeUp_GMat = cv2.gapi.resize(blur_GMat, (self.w*self.multiplier, self.h*self.multiplier))
        thresh_GMat, thresh = cv2.gapi.threshold(resizeUp_GMat, maxval_sc, cv2.THRESH_OTSU)
        resizeDown_GMat = cv2.gapi.resize(thresh_GMat, (self.w, self.h))

        self.comp = cv2.GComputation(cv2.GIn(img_GMat, maxval_sc), cv2.GOut(resizeDown_GMat, thresh))

    def apply(self): 
        # compute 
        mat, thresh = self.comp.apply(cv2.gin(self.img, 255), args=cv2.gapi.compile_args(self.ocl_kernel))
        return "ok"

img = cv2.imread("lenna.png")
pipeline = Pipeline(img)
pipeline.graph()

times = []
for i in range (5000):
    e1 = cv2.getTickCount()
    pipeline.apply()
    e2 = cv2.getTickCount()
    times.append((e2 - e1)/ cv2.getTickFrequency())

time_avg = np.array(times).mean()

print("Execution Time : %.4f (FPS %.2f)" % (time_avg, (1/time_avg)))