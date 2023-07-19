/**
MIT License

Copyright (c) 2018 NVIDIA CORPORATION. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*
*/

#include "ds_image.h"
#include "trt_utils.h"
#include "yolo_config_parser.h"
#include "yolov3.h"

#include <cstdio>
#include <csignal>
#include <cstdlib>
#include <unistd.h>
#include <queue>
#include <thread>
#include <sys/time.h>
#include <opencv2/videoio.hpp>

std::queue<cv::Mat> readQueue;
std::queue<cv::Mat> writeQueue;

volatile sig_atomic_t stop = 0;

void sigint_handler(int s){
    printf("\nCleaning resources...\n");
    stop = 1;
}

void readFrame(cv::VideoCapture& cap) {
    cv::Mat frame;

    while (!stop)
    {
        cap >> frame;

        if (frame.empty()) {
            std::cout << "Could not read camera" << std::endl;
            return;
        }

        readQueue.push(frame);
    }
}

void processFrame(std::unique_ptr<Yolo>& inferNet) {
    DsImage dsImage;
    while (!stop