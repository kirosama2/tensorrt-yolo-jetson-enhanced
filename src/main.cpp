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
    while (!stop)
    {
        if (readQueue.empty()){
            continue;
        }

        cv::Mat frame = readQueue.front();
        readQueue.pop();


        // Load a new batch
        dsImage = DsImage(frame, inferNet->getInputH(), inferNet->getInputW());
        cv::Mat trtInput = blobFromDsImage(dsImage, inferNet->getInputH(), inferNet->getInputW());

        // struct timeval inferStart, inferEnd;
        // gettimeofday(&inferStart, NULL);

        inferNet->doInference(trtInput.data, 1);

        // gettimeofday(&inferEnd, NULL);
        // double inferElapsed = ((inferEnd.tv_sec - inferStart.tv_sec)
        //                 + (inferEnd.tv_usec - inferStart.tv_usec) / 1000000.0)
        //                 * 1000;
        // std::cout << "Frame process time: " << inferElapsed << "ms" << std::endl;

        auto binfo = inferNet->decodeDetections(0, dsImage.getImageHeight(),
                                                dsImage.getImageWidth());
        auto remaining
            = nmsAllClasses(inferNet->getNMSThresh(), binfo, inferNet->getNumClasses());
        for (auto b : remaining)
        {
            printPredictions(b, inferNet->getClassName(b.label));
            dsImage.addBBox(b, inferNet->getClassName(b.label));
        }

        cv::Mat img = dsImage.getMaskedImage();

        writeQueue.push(img);
    }
}

void writeFrame(cv::VideoWriter& out) {
    while (!stop) {
        if (writeQueue.empty()) {
            continue;
        }

        cv::Mat img = writeQueue.front();
        writeQueue.pop();

        // struct timeval inferStart, inferEnd;
        // gettimeofday(&inferStart, NULL);

        out << img;

        // gettimeofday(&inferEnd, NULL);
        // double inferElapsed = ((inferEnd.tv_sec - inferStart.tv_sec)
        //                  + (inferEnd.tv_usec - inferStart.tv_usec) / 1000000.0)
        //                  * 1000;
        // std::cout << "Frame write time: " << inferElapsed << "ms" << std::endl;
    }
}

int main(int argc, char** argv)
{
    // Flag set in the command line overrides the value in the flagfile
    gflags::SetUsageMessage(
        "Usage : trt-yolo-app --flagfile=</path/to/config_file.txt> --<flag>=value ...");

    // parse config params
    yoloConfigParserInit(argc, argv);
    NetworkInfo yoloInfo = getYoloN