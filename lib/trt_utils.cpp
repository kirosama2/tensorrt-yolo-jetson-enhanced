
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

#include "trt_utils.h"

#include <experimental/filesystem>
#include <fstream>
#include <iomanip>

cv::Mat blobFromDsImage(const DsImage& inputImage, const int& inputH,
                         const int& inputW)
{
    std::vector<cv::Mat> letterboxStack(1);
    inputImage.getLetterBoxedImage().copyTo(letterboxStack.at(0));

    return cv::dnn::blobFromImages(letterboxStack, 1.0, cv::Size(inputW, inputH),
                                   cv::Scalar(0.0, 0.0, 0.0), false, false);
}

cv::Mat blobFromDsImages(const std::vector<DsImage>& inputImages, const int& inputH,
                         const int& inputW)
{
    std::vector<cv::Mat> letterboxStack(inputImages.size());
    for (uint i = 0; i < inputImages.size(); ++i)
    {
        inputImages.at(i).getLetterBoxedImage().copyTo(letterboxStack.at(i));
    }
    return cv::dnn::blobFromImages(letterboxStack, 1.0, cv::Size(inputW, inputH),
                                   cv::Scalar(0.0, 0.0, 0.0), false, false);
}

static void leftTrim(std::string& s)
{
    s.erase(s.begin(), find_if(s.begin(), s.end(), [](int ch) { return !isspace(ch); }));
}

static void rightTrim(std::string& s)
{
    s.erase(find_if(s.rbegin(), s.rend(), [](int ch) { return !isspace(ch); }).base(), s.end());
}

std::string trim(std::string s)
{
    leftTrim(s);
    rightTrim(s);
    return s;
}

float clamp(const float val, const float minVal, const float maxVal)
{
    assert(minVal <= maxVal);
    return std::min(maxVal, std::max(minVal, val));
}

bool fileExists(const std::string fileName, bool verbose)
{
    if (!std::experimental::filesystem::exists(std::experimental::filesystem::path(fileName)))
    {
        if (verbose) std::cout << "File does not exist : " << fileName << std::endl;
        return false;
    }
    return true;
}

BBox convertBBoxNetRes(const float& bx, const float& by, const float& bw, const float& bh,
                       const uint& stride, const uint& netW, const uint& netH)
{
    BBox b;
    // Restore coordinates to network input resolution
    float x = bx * stride;
    float y = by * stride;

    b.x1 = x - bw / 2;
    b.x2 = x + bw / 2;

    b.y1 = y - bh / 2;