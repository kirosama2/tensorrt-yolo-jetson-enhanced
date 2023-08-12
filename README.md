# Enhanced Video YOLO with TensorRT on Jetson Nano

This is a customized version of the [Jetson Nano: Deep Learning Inference Benchmarks Instructions](https://devtalk.nvidia.com/default/topic/1050377/jetson-nano/deep-learning-inference-benchmarking-instructions/).

Our goal is to run real-time object detections on Jetson Nano with TensorRT optimized YOLO network.

## Evaluation

|   Model   | Frame Rate |
| :-------: | :--------: |
|   YOLOv3  |  2 - 5 fps |
| YOLOv3-tiny |    24 fps |

Although YOLOv3-tiny runs much faster, the detection result is not as good.

## Setup

Ensure you have the necessary prerequisites, and fetch the weights:

```bash
chmod +x prebuild.sh
./prebuild.sh
```

## Compile & Run

```bash

cd src
mkdir build && cd build
cmake -D CMAKE_BUILD_TYPE=Release ..
make
cd ../../

./src/build/trt-yolo-ap