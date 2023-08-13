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

./src/build/trt-yolo-app --flagfile=/path/to/config-file.txt

# e.g.

./src/build/trt-yolo-app --flagfile=config/yolov3-tiny.txt
```

Refer to sample config files `yolov2.txt`, `yolov2-tiny.txt`, `yolov3.txt` and `yolov3-tiny.txt` in `config/` directory.

### GStreamer Reference

This project leverages `GStreamer` to make full use of Nvidia's hardware acceleration on video capturing, encoding, and decoding. Adapt the code in `src/main.cpp` to match with your Jetson Nano camera setup.

For instance, for testing, I used the following on a Jetson Nano:

```bash
