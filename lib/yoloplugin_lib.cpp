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

#include "yoloplugin_lib.h"
#include "yolo_config_parser.h"
#include "yolov2.h"
#include "yolov3.h"

#include <iomanip>
#include <sys/time.h>

static void decodeBatchDetections(const YoloPluginCtx* ctx, std::vector<YoloPluginOutput*>& outputs)
{
    for (uint p = 0; p < ctx->batchSize; ++p)
    {
        YoloPluginOutput* out = new YoloPluginOutput;
        std::vector<BBoxInfo> binfo = ctx->inferenceNetwork->decodeDetections(
            p, ctx->initParams.processingHeight, ctx->initParams.processingWidth);
        std::vector<BBoxInfo> remaining = nmsAllClasses(
            ctx->inferenceNetwork->getNMSThresh(), binfo, ctx->inferenceNetwork->getNumClasses());
        out->numObjects = remaining.size();
        assert(out->numObjects <= MAX_OBJECTS_PER_FRAME);
        for (uint j = 0; j < remaining.size(); ++j)
        {
            BBoxInfo b = remaining.at(j);
            YoloPluginObject obj;
            obj.left = static_cast<int>(b.box.x1);
            obj.top = static_cast<int>(b.box.y1);
            obj.width = static_cast<int>(b.box.x2 - b.box.x1);
            obj.height = static_cast<int>(b.box.y2 - b.box.y1);
            strcpy(obj.label, ctx->inferenceNetwork->getClassName(b.label).c_str());
            out->object[j] = obj;

            if (ctx->inferParams.printPredictionInfo)
            {
                printPredictions(b, ctx->inferenceNetwork->getClassName(b.label));
            }
        }
        outputs.at(p) = out;
    }
}

static void dsPreProcessBatchInput(const std::vector<cv::Mat*>& cvmats, cv::Mat& batchBlob,
                                   const int& processingHeight, const int& processingWidth,
                                   const int& inputH, const int& inputW)
{

    std::vector<cv::Mat> batch_images(
        cvmats.size(), cv::Mat(cv::Size(processingWidth, processingHeight), CV_8UC3));
    for (uint i = 0; i < cvmats.size(); ++i)
    {
        cv::Mat imageResize, imageBorder, inputImage;
        inputImage = *cvmats.at(i);
        int maxBorder = std::max(inputImage.size().width, inputImage.size().h