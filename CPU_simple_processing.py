import cv2
import numpy as np 

class Pipeline: 
    def __init__(self, img):
        self.multiplier = 2
        self.h, self.w, self.c = img.shape
        self.img = img
        self.gray = np.zeros(img.shape[:2], np.uint8)
        self.blur = np.zeros((self.h, self.w), np.uint8)
        self.resizeUp = np.zeros((self.h*self.multiplier, self.w*self.multiplier), np.uint8)
        self.threshold = np.zeros((self.h*self.multiplier, self.w*self.multiplier), np.uint8)
        self.resizeDown_Umat = np.zeros((self.h, self.w), np.uint8)

    def apply(self):
        cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY, self.gray)
        # cv2.medianBlur(self.gray, 3, self.blur)
        # cv2.resize(self.blur, (self.w*self.multiplier, self.h*self.multiplier), self.resizeUp)
        cv2.resize(self.gray, (self.w*self.multiplier, self.h*self.multiplier), self.resizeUp)
        self.threshold, thresh = cv2.threshold(self.resizeUp, 0, 255, cv2.THRESH_OTSU)
        cv2.resize(self.threshold, (self.w, self.h), self.resizeDown_Umat)
        
        return self.resizeDown_Umat, thresh
            

img = cv2.imread("lenna.png")
pipeline = Pipeline(img)

times = []
for i in range (300):
    e1 = cv2.getTickCount()
    mat, thresh = pipeline.apply()
    e2 = cv2.getTickCount()
    times.append((e2 - e1)/ cv2.getTickFrequency())

time_avg = np.array(times).mean()

print("Execution Time : %.4f (FPS %.2f)" % (time_avg, (1/time_avg)))