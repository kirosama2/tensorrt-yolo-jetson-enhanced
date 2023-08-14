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
gst-launch-1.0 nvarguscamerasrc ! \
    'video/x-raw(memory:NVMM), width=(int)1920, height=(int)1080, format=(string)NV12, framerate=(fraction)30/1' ! \
    nvvidconv flip-method=0 ! 'video/x-raw, format=(string)BGRx' ! \
    videoconvert ! 'video/x-raw, format=(string)BGR' ! \
    videoconvert ! 'video/x-raw, format=(string)RGB' ! \
    videoconvert ! nvvidconv ! 'video/x-raw(memory:NVMM), format=(string)NV12' ! \
    nvv4l2h265enc insert-sps-pps=true ! 'video/x-h265, stream-format=(string)byte-stream' ! \
    queue ! h265parse ! queue ! \
    rtph265pay ! queue ! \
    udpsink host=192.168.1.194 port=1234
```

The command above reads frames from the camera and sends the stream to `192.168.1.194`, which is my desktop address in LAN.

From desktop, execute:

```bash
gst-launch-1.0 udpsrc port=1234 ! \
    application/x-rtp,encoding-name=H265 ! queue ! \
    rtph265depay ! queue ! avdec_h265 ! queue ! autovideosink
```

This allows you to view the streaming video on your desktop, which is being captured on Jetson Nano.

This means the GStreamer pipeline is functional, so these commands cou