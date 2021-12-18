import cv2 
import numpy as np 

import cv2
import numpy as np 

class Pipeline: 
    def __init__(self, img):
        self.h, self.w, self.c = img.shape
        self.img = img

        modelConfiguration = "yolo/coco_yolov3-tiny.cfg"
        modelWeights = "yolo/coco_yolov3-tiny.weights"
        self.net = cv2.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
        self.net.getLayerNames()
        self.layerOutput = self.net.getUnconnectedOutLayersNames()

        #self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        #self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        #self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_OPENCL)
        #self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_OPENCL_FP16)

        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
        #self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)

        self.target_w = 416
        self.target_h = 416

        self.classes = None
        self.classesFile = "yolo/coco.names"
        with open(self.classesFile, 'rt') as f:
            self.classes = f.read().rstrip('\n').split('\n')  

        self.blob = None
        print("[INFO] model loaded...")

    def apply(self):
        self.blob = cv2.dnn.blobFromImage(self.img, 
                            1.0/255, 
                            (self.target_w, self.target_h), 
                            (0, 0, 0), swapRB=True, crop=False)

        # Forward pass
        self.net.setInput(self.blob)
        output = self.net.forward(self.layerOutput)
        
        return output
            

img = cv2.imread("lenna.png")
pipeline = Pipeline(img)

times = []
for i in range (100):
    e1 = cv2.getTickCount()
    mat, thresh = pipeline.apply()
    e2 = cv2.getTickCount()
    times.append((e2 - e1)/ cv2.getTickFrequency())

time_avg = np.array(times).mean()

print("Execution Time : %.4f (FPS %.2f)" % (time_avg, (1/time_avg)))