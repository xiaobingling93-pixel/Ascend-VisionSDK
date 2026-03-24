/*
 * -------------------------------------------------------------------------
 *  This file is part of the Vision SDK project.
 * Copyright (c) 2026 Huawei Technologies Co.,Ltd.
 *
 * Vision SDK is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *
 *           http://license.coscl.org.cn/MulanPSL2
 *
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 * -------------------------------------------------------------------------
 */

#include <fstream>
#include <algorithm>
#include <opencv2/opencv.hpp>
#include "MxBase/E2eInfer/ImageProcessor/ImageProcessor.h"
#include "MxBase/MxBase.h"
#include "MxBase/E2eInfer/Tensor/Tensor.h"

using namespace MxBase;

void TestYuvsp420ToTensorSuccess(int deviceId)
{
    std::string imagePath = "/home/vision_data/Tensortest.jpg";
    cv::Mat inputImage = cv::imread(imagePath);
    if (inputImage.empty()) {
        std::cout << "Failed to read image: " << imagePath << std::endl;
        exit(1);
    }
    int newWidth = inputImage.cols + (inputImage.cols % 2);
    int newHeight = inputImage.rows + (inputImage.rows % 2);

    cv::Mat resizedImage;
    cv::resize(inputImage, resizedImage, cv::Size(newWidth, newHeight));
    cv::Mat yuvImage;
    cv::cvtColor(resizedImage, yuvImage, cv::COLOR_BGR2YUV_I420);
    cv::cvtColor(yuvImage, yuvImage, cv::COLOR_YUV2BGR_NV12);
    Size size(resizedImage.cols, resizedImage.rows);
    size_t dataSize = yuvImage.total() * yuvImage.elemSize();

    std::shared_ptr<uint8_t> imageData(new uint8_t[dataSize], std::default_delete<uint8_t[]>());
    std::copy(yuvImage.data, yuvImage.data + dataSize, imageData.get());

    Image image(imageData, dataSize, -1, size, ImageFormat::YUV_SP_420);
    Tensor tensor = image.ConvertToTensor(true, true);
    if (tensor.IsEmpty()) {
        std::cout << "Failed to ConvertToTensor in TestYuvsp420ToTensorSuccess." << std::endl;
        exit(1);
    }
}

void TestYuvsp422ToTensorSuccess(int deviceId)
{
    constexpr int width = 640;
    constexpr int height = 640;
    constexpr size_t dataSize = width * height * 4;

    std::shared_ptr<uint8_t> data(new uint8_t[dataSize]);
    for (size_t i = 0; i < dataSize; ++i) {
        data.get()[i] = 15; // Fill with test data
    }

    Size size(width, height);
    Image image(data, dataSize, -1, size, ImageFormat::YUV_SP_422);
    Tensor tensor = image.ConvertToTensor(true, true);
    if (tensor.IsEmpty()) {
        std::cout << "Failed to ConvertToTensor in TestYuvsp420ToTensorSuccess." << std::endl;
        exit(1);
    }
}

int main(int argc, char *argv[])
{
    if (argc != 2) {
        std::cout << "There should be two parameters." << std::endl;
        return 1;
    }

    std::string functionName = argv[1];
    constexpr int deviceId = 0;
    APP_ERROR ret = MxInit();
    if (ret != APP_ERR_OK) {
        std::cout << "MxInit failed" << std::endl;
        return 1;
    }
    {
        if (functionName == "TestYuvsp420ToTensorSuccess") {
            TestYuvsp420ToTensorSuccess(deviceId);
        } else if (functionName == "TestYuvsp422ToTensorSuccess") {
            TestYuvsp422ToTensorSuccess(deviceId);
        } else {
            std::cout << "Unknown function: " << functionName << std::endl;
            exit(1);
        }
    }
    MxDeInit();
    return 0;
}