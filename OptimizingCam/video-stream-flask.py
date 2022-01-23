import os
import cv2
import numpy as np 
from gst_cam import camera
from flask import Flask, render_template, Response

# host mem not implemented, manually pin memory
class PinnedMem(object):
    def __init__(self, h, w, dtype=cv2.CV_8UC3):
        self.mem            = cv2.cuda_HostMem(h, w, dtype, cv2.cuda.HostMem_PAGE_LOCKED)
        self.array          = self.mem.createMatHeader()
        self.pinned         = True

    def __del__(self):
        cv2.cuda.unregisterPageLocked(self.array)
        self.pinned         = False
        
    def __repr__(self):
        return f'pinned = {self.pinned}'


class Pipeline: 
    def __init__(self, w=1280, h=720, multiplier=2, lr=0.05, n_streams = 2):
        self.cap            = cv2.VideoCapture(camera(0, w, h, fs=60), cv2.CAP_GSTREAMER)
        self.w, self.h      = w, h
        self.multiplier     = multiplier
        self.mog            = cv2.cuda.createBackgroundSubtractorMOG2()
        self.lr             = lr

        self.n_streams      = n_streams
        self.streams        = []
        self.stream_index   = 0
        self.init_stream()  # CUDA Stream initialization

        self.imgs_in        = []
        self.imgs           = []
        self.grays          = []
        self.resizeUps      = []
        self.mog_imgs       = []
        self.resizeDowns    = []
        self.result_imgs    = []
        self.memory_index   = 0
        self.init_memory()  # memory allocation

        self.is_next_frame  = False

    def init_memory(self):
        self.imgs_in        = [PinnedMem(self.h, self.w) for __ in range(self.n_streams + 1)]

        for __ in range(self.n_streams) :
            self.imgs.append(cv2.cuda_GpuMat(self.h, self.w, cv2.CV_8UC3))
            self.grays.append(cv2.cuda_GpuMat(self.h, self.w, cv2.CV_8UC1))
            self.resizeUps.append(cv2.cuda_GpuMat(self.h*self.multiplier, self.w*self.multiplier, cv2.CV_8UC1))
            self.mog_imgs.append(cv2.cuda_GpuMat(self.h*self.multiplier, self.w*self.multiplier, cv2.CV_8UC1))
            self.resizeDowns.append(cv2.cuda_GpuMat(self.h, self.w, cv2.CV_8UC1))
            self.result_imgs.append(PinnedMem(self.h, self.w, dtype=cv2.CV_8UC1))

    def init_stream(self): 
        self.streams        = [cv2.cuda_Stream() for __ in range(self.n_streams)]

    def apply(self):
        ret, __             = self.cap.read(self.getFrame())
        if not ret : 
            raise Exception("Invalid image frame!")

        i = self.stream_index
        self.setNextStream()

        if (self.is_next_frame) : 
            self.streams[i].waitForCompletion() # wait after we have read the next frame
        else : 
            self.is_next_frame = True

        self.imgs[i].upload(self.imgs_in[self.memory_index].array, stream=self.streams[i])
        cv2.cuda.cvtColor(self.imgs[i], cv2.COLOR_BGR2GRAY, self.grays[i], stream=self.streams[i])
        cv2.cuda.resize(self.grays[i], (self.w*self.multiplier, self.h*self.multiplier), self.resizeUps[i], stream=self.streams[i])
        self.mog.apply(self.resizeUps[i], fgmask=self.mog_imgs[i], learningRate = self.lr, stream=self.streams[i])
        cv2.cuda.resize(self.mog_imgs[i], (self.w, self.h), self.resizeDowns[i], stream=self.streams[i])
        self.resizeDowns[i].download(self.streams[i], self.result_imgs[i].array)
        self.streams[i].queryIfComplete()

        return i
            
    def getFrame(self):
        self.memory_index   = (self.memory_index + 1) % len(self.imgs_in)
        return self.imgs_in[self.memory_index].array

    def setNextStream(self):
        self.stream_index   = (self.stream_index + 1) % self.n_streams

    def sync(self):
        for __ in range(self.n_streams):
            if not self.streams[self.stream_index].queryIfComplete() :
                self.streams[self.stream_index].waitForCompletion()

    def close(self): 
        self.cap.release()

    def gen_frames(self):  
        while True:
            yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + 
                cv2.imencode('.jpg', self.result_imgs[self.apply()].array)[1].tobytes() + 
                b'\r\n')

# create pipeline
pipeline    = Pipeline()

app         = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(pipeline.gen_frames(), 
                    mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == "__main__" :
    app.run(host="0.0.0.0")
    pipeline.sync()
    pipeline.close()