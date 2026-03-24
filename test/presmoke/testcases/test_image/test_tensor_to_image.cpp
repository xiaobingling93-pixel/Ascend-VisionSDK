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

#include <opencv2/opencv.hpp>
#include "MxBase/E2eInfer/ImageProcessor/ImageProcessor.h"
#include "MxBase/MxBase.h"
#include "MxBase/E2eInfer/Tensor/Tensor.h"

using namespace cv;
using namespace std;
using namespace MxBase;

Image TensorToImage(const Tensor &inputTensor, const ImageFormat &format)
{
    Image outputImage;
    APP_ERROR result = Image::TensorToImage(inputTensor, outputImage, format);
    if (result != APP_ERR_OK) {
        std::cout << "TensorToImage failed." << std::endl;
        MxBase::MxDeInit();
        exit(1);
    }
    return outputImage;
}

void ImageEncode(ImageProcessor &imageProcessor, const Image &inputImage, const std::string &outputPath)
{
    constexpr uint32_t encodeQuality = 100;
    APP_ERROR result = imageProcessor.Encode(inputImage, outputPath, encodeQuality);
    if (result != APP_ERR_OK) {
        std::cout << "ImageProcessor Encode failed." << std::endl;
        MxBase::MxDeInit();
        exit(1);
    }
}

void TestTensorToImageFromImageSuccess(int deviceId, ImageProcessor &imageProcessor)
{
    const std::string inputImagePath = "/home/vision_data/Tensortest.jpg";
    const std::string outputImagePath = "tensorToimage.jpg";

    Image outputImage;
    cv::Mat convertedMat;
    cv::Mat inputMat = cv::imread(inputImagePath, IMREAD_UNCHANGED);
    cv::cvtColor(inputMat, convertedMat, cv::COLOR_BGR2RGB);
    uint32_t channels = convertedMat.channels();
    uint32_t height = convertedMat.rows;
    uint32_t width = convertedMat.cols;
    std::vector<uint32_t> hwcTensorShape = {height, width, channels};

    MxBase::Tensor inputTensor(convertedMat.data, hwcTensorShape, MxBase::TensorDType::UINT8);
    inputTensor.ToDevice(deviceId);
    outputImage = TensorToImage(inputTensor, ImageFormat::RGB_888);
    ImageEncode(imageProcessor, outputImage, outputImagePath);
}

void Test2DimTensorToYuvSuccess(int deviceId, ImageProcessor &imageProcessor)
{
    Image outputImage;
    std::vector<uint32_t> twoDimTensorShape = {2, 2};
    uint8_t tensorData[] = {255, 2, 255, 4};
    MxBase::Tensor inputTensor(&tensorData, twoDimTensorShape, TensorDType::UINT8);
    inputTensor.ToDevice(deviceId);
    outputImage = TensorToImage(inputTensor, ImageFormat::YUV_400);
}

void TestUint32TensorToRgbFailed(int deviceId, ImageProcessor &imageProcessor)
{
    Image outputImage;
    std::vector<uint32_t> threeDimTensorShape = {3, 4, 1};
    uint32_t tensorData[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    MxBase::Tensor inputTensor(&tensorData, threeDimTensorShape, TensorDType::UINT32);
    inputTensor.ToDevice(deviceId);
    outputImage = TensorToImage(inputTensor, ImageFormat::RGB_888);
}

void Test2DimTensorToBgrFailed(int deviceId, ImageProcessor &imageProcessor)
{
    Image outputImage;
    std::vector<uint32_t> twoDimTensorShape = {2, 2};
    uint8_t tensorData[] = {255, 2, 255, 4};
    MxBase::Tensor inputTensor(&tensorData, twoDimTensorShape, TensorDType::UINT8);
    inputTensor.ToDevice(deviceId);
    outputImage = TensorToImage(inputTensor, ImageFormat::BGR_888);
}

int main(int argc, char *argv[])
{
    constexpr int paraNum = 2;
    constexpr int deviceId = 0;
    if (argc != paraNum) {
        std::cout << "There should be two parameters." << std::endl;
        return 1;
    }
    std::string functionName = argv[1];

    APP_ERROR ret = MxBase::MxInit();
    if (ret != APP_ERR_OK) {
        std::cout << "MxInit failed." << std::endl;
        return 1;
    }
    {
        ImageProcessor imageProcessor(deviceId);

        if (functionName == "TestTensorToImageFromImageSuccess") {
            TestTensorToImageFromImageSuccess(deviceId, imageProcessor);
        } else if (functionName == "Test2DimTensorToYuvSuccess") {
            Test2DimTensorToYuvSuccess(deviceId, imageProcessor);
        } else if (functionName == "TestUint32TensorToRgbFailed") {
            TestUint32TensorToRgbFailed(deviceId, imageProcessor);
        } else if (functionName == "Test2DimTensorToBgrFailed") {
            Test2DimTensorToBgrFailed(deviceId, imageProcessor);
        } else {
            std::cout << "Error: Unknown function name: " << functionName << std::endl;
            exit(1);
        }
    }
    MxBase::MxDeInit();
    return 0;
}