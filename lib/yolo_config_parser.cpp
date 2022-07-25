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
DEFINE_bool(print_perf_info, false, "[OPTIONAl] Print per