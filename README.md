# OpenCV Benchmark

## Simple Image Processing
- Native OpenCV (CPU) : Raspberry Pi 3, Jetson TK1 & Jetson Nano
- OpenCV T-API (OpenCL Backend) : Raspberry Pi 3, Jetson TK1 & Jetson Nano
- OpenCV G-API (CPU Kernel) : Raspberry Pi 3, Jetson TK1 & Jetson Nano
- OpenCV G-API (OpenCL Kernel) : Raspberry Pi 3, Jetson TK1 & Jetson Nano

## Deep Learning Inferencing (Tiny Yolo V3 416x416)
- OpenCV DNN (Backend OpenCV, Target CPU) : Raspberry Pi 3, Jetson TK1 & Jetson Nano
- OpenCV DNN (Backend OpenCV, Target OpenCL) : Raspberry Pi 3, Jetson TK1 & Jetson Nano
- OpenCV DNN (Backend CUDA, Target CUDA) : Jetson Nano
- G-API Inference (CPU Kernel) : Raspberry Pi 3, Jetson TK1 & Jetson Nano
- G-API Inference (OpenCL Kernel) : Raspberry Pi 3, Jetson TK1 & Jetson Nano

## Additional Note
- Download `yolov3.weights` from [google drive](https://drive.google.com/file/d/1NrC8t0_QgFkv1ZH57TN4wLgycNMu4m6i/view?usp=sharing).
- Raspberry Pi 3 : 
    - Maximize CPU clock into 1.2GHz
    - Maximize dram clock into 600MHz
- Jetson Nano : 
    - Maximize Jetson Performance : 
        - `sudo nvpmodel -m 0`
        - `sudo jetson_clocks`

# Benchmark Result 
## Raspberry Pi 3 (OpenCV Simple Processing)
- OpenCV CPU : Execution Time : 0.0320 (FPS 31.26)
- T-API (OpenCL Backend) : Execution Time : 0.0313 (FPS 31.90)
- G-API (OpenCL Kernel) : Execution Time : 0.0346 (FPS 28.89)
- G-API (CPU Kernel) : Execution Time : 0.0297 (FPS 33.63)

## Raspberry Pi 3 (OpenCV Deep Learning Inferencing)
- OpenCV DNN (Backend OpenCV, Target CPU) : Execution Time : 2.1880 (FPS 0.46)
- OpenCV DNN (Backend OpenCV, Target OpenCL) : ❌ fall back to CPU.
- G-API Inference (OpenCL Kernel) : `TODO`
- G-API Inference (CPU Kernel) : `TODO`


## Jetson Nano (OpenCV Simple Processing)
- OpenCV CPU : Execution Time : 0.0050 (FPS 201.16)
- T-API (OpenCL Backend) : Execution Time : 0.0063 (FPS 159.32)
- G-API (OpenCL Kernel) : ❌ core dumped (CUDA_ERROR_NOT_SUPPORTED)
- G-API (CPU Kernel) : Execution Time : 0.0068 (FPS 146.32)

## Jetson Nano (OpenCV Deep Learning Inferencing)
- OpenCV DNN (Backend OpenCV, Target CPU) : Execution Time : 0.3618 (FPS 2.76)
- OpenCV DNN (Backend OpenCV, Target OpenCL) : ❌ core dumped (CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES)
- OpenCV DNN (Backend CUDA, Target CUDA) : Execution Time : 0.2826 (FPS 3.54)
- G-API Inference (OpenCL Kernel) : `TODO`
- G-API Inference (CPU Kernel) : `TODO`

## Jetson Nano MAX mode (OpenCV Simple Processing)
- OpenCV CPU : Execution Time : 0.0049 (FPS 203.71)
- T-API (OpenCL Backend) : Execution Time : 0.0057 (FPS 175.19)
- G-API (OpenCL Kernel) : ❌ core dumped (CUDA_ERROR_NOT_SUPPORTED)
- G-API (CPU Kernel) : Execution Time : 0.0068 (FPS 146.95)

## Jetson Nano MAX mode (OpenCV Deep Learning Inferencing)
- OpenCV DNN (Backend OpenCV, Target CPU) : Execution Time : 0.3562 (FPS 2.81)
- OpenCV DNN (Backend OpenCV, Target OpenCL) : ❌ core dumped (CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES)
- OpenCV DNN (Backend CUDA, Target CUDA) : Execution Time : 0.1699 (FPS 5.89)
- G-API Inference (OpenCL Kernel) : `TODO`
- G-API Inference (CPU Kernel) : `TODO`

## Jetson TK1 (OpenCV Simple Processing)
- OpenCV CPU : Execution Time : 0.0118 (FPS 84.48)
- T-API (OpenCL Backend) : ❌ aborted (pocl-cuda: failed to generate PTX)
- G-API (OpenCL Kernel) : ❌ aborted (pocl-cuda: failed to generate PTX)
- G-API (CPU Kernel) : Execution Time : 0.0105 (FPS 94.85)

## Jetson TK1 (OpenCV Deep Learning Inferencing)
- OpenCV DNN (Backend OpenCV, Target CPU) : Execution Time : 0.7818 (FPS 1.28)
- OpenCV DNN (Backend OpenCV, Target OpenCL) : ❌ aborted (pocl-cuda: failed to generate PTX)
- OpenCV DNN (Backend CUDA, Target CUDA) : ❌ OpenCV DNN backend CUDA not supported in Jetson TK1
- G-API Inference (OpenCL Kernel) : `TODO`
- G-API Inference (CPU Kernel) : `TODO`