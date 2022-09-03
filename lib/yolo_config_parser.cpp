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

#include "yolo_config_parser.h"

#include <assert.h>
#include <iostream>

DEFINE_string(network_type, "not-specified",
              "[REQUIRED] Type of network architecture. Choose from yolov2, yolov2-tiny, "
              "yolov3 and yolov3-tiny");
DEFINE_string(config_file_path, "not-specified", "[REQUIRED] Darknet cfg file");
DEFINE_string(wts_file_path, "not-specified", "[REQUIRED] Darknet weights file");
DEFINE_string(labels_file_path, "not-specified", "[REQUIRED] Object class labels file");
DEFINE_string(precision, "kFLOAT",
              "[OPTIONAL] Inference precision. Choose from kFLOAT, kHALF and kINT8.");
DEFINE_string(deviceType, "kGPU",
              "[OPTIONAL] The device that this layer/network will execute on. Choose from kGPU and kDLA(only for kHALF).");
DEFINE_string(calibration_table_path, "not-specified",
              "[OPTIONAL] Path to pre-generated calibration table. If flag is not set, a new calib "
              "table <network-type>-<precision>-calibration.table will be generated");
DEFINE_string(engine_file_path, "not-specified",
              "[OPTIONAL] Path to pre-generated engine(PLAN) file. If flag is not set, a new "
              "engine <network-type>-<precision>-<batch-size>.engine will be generated");
DEFINE_string(input_blob_name, "data",
              "[OPTIONAL] Name of the input layer in the tensorRT engine file");
DEFINE_bool(print_perf_info, false, "[OPTIONAl] Print performance info on the console");
DEFINE_bool(print_prediction_info, false, "[OPTIONAL] Print detection info on the console");
DEFINE_string(
    test_images, "data/test_images.txt",
    "[REQUIRED] Text file containing absolute paths or filenames of all the images to be "
    "used for inference. If only filenames are provided, their corresponding source directory "
    "has to be provided through 'test_images_path' flag");
DEFINE_string(test_images_path, "not-specified",
              "[OPTIONAL] absolute source directory path of the list of images supplied through "
              "'test_images' flag");
DEFINE_string(calibration_images, "data/calibration_images.txt",
              "[OPTIONAL] Text file containing absolute paths or filenames of calibration images. "
              "Flag required if precision is kINT8 and there is not pre-generated calibration "
              "table. If only filenames are provided, their corresponding source directory has to "
              "be provided through 'calibration_images_path' flag");
DEFINE_string(calibration_images_path, "not-specified",
              "[OPTIONAL] absolute source directory path of the list of images supplied through "
              "'calibration_images' flag");
DEFINE_uint64(batch_size, 1, "[OPTIONAL] Batch size for the inference engine.");
DEFINE_double(prob_thresh, 0.5, "[OPTIONAL] Probability threshold for detected objects");
DEFINE_double(nms_thresh, 0.5, "[OPTIONAL] IOU threshold for bounding box candidates");
DEFINE_bool(do_benchmark, false,
            "[OPTIONAL] Generate JSON file with detection info in coco benchmark format");
DEFINE_bool(save_detections, false,
            "[OPTIONAL] Flag to save images overlayed with objects detected.");
DEFINE_bool(view_detections, false,
            "[OPTIONAL] Flag to view images overlayed with objects detected.");
DEFINE_string(save_detections_path, "not-specified",
              "[OPTIONAL] Path where the images overlayed with bounding boxes are to be saved");
DEFINE_bool(
    decode, true,
    "[OPTIONAL] Decode the detections. This can be set to false if benchmarking network for "
    "throughput only");
DEFINE_uint64(seed, std::time(0), "[OPTIONAL] Seed for the random number generator");
DEFINE_bool(shuffle_test_set, false,
            "[OPTIONAL] Shuffle the test set images before running inference");

static bool isFlagDefault(std::string flag) { return flag == "not-specified" ? true : false; }

static bool networkTypeValidator(const char* flagName, std::string value)
{
    if (((FLAGS_network_type) == "yolov2") || ((FLAGS_network_type) == "yolov2-tiny")
        || ((FLAGS_network_type) == "yolov3") || ((FLAGS_network_type) == "yolov3-tiny"))
        return true;

    else
        std::cout << "Invalid value for --" << flagName << ": " << value << std::endl;

    return false;
}

static bool precisionTypeValidator(const char* flagName, std::string value)
{
    if ((FLAGS_precision == "kFLOAT") || (FLAGS_precision == "kINT8")
        || (FLAGS_precision == "kHALF"))
        return true;
    else
        std::cout << "Invalid value for --" << flagName << ": " << value << std::endl;
    return false;
}

static bool verifyRequiredFlags()
{
    asse