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

#include <cstdint>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include "MxBase/E2eInfer/ImageProcessor/ImageProcessor.h"
#include "MxBase/MxBase.h"
#include "MxBase/E2eInfer/Tensor/Tensor.h"

using namespace cv;
using namespace std;
using namespace MxBase;

namespace {

constexpr int PARA_NUM = 2;
constexpr int DEVICE_ID = 0;
constexpr char INPUT_IMAGE_PATH[] = "/home/vision_data/Tensortest.jpg";

[[noreturn]] void ExitCaseFailure()
{
    MxBase::MxDeInit();
    exit(1);
}

cv::Mat LoadRgbImage()
{
    cv::Mat inputMat = cv::imread(INPUT_IMAGE_PATH, IMREAD_UNCHANGED);
    if (inputMat.empty()) {
        std::cout << "Failed to read image: " << INPUT_IMAGE_PATH << std::endl;
        ExitCaseFailure();
    }

    cv::Mat convertedMat;
    if (inputMat.channels() == 4) {
        cv::cvtColor(inputMat, convertedMat, cv::COLOR_BGRA2RGB);
    } else if (inputMat.channels() == 3) {
        cv::cvtColor(inputMat, convertedMat, cv::COLOR_BGR2RGB);
    } else if (inputMat.channels() == 1) {
        cv::cvtColor(inputMat, convertedMat, cv::COLOR_GRAY2RGB);
    } else {
        convertedMat = inputMat;
    }
    return convertedMat;
}

size_t GetElementCount(const std::vector<uint32_t> &shape)
{
    size_t count = 1;
    for (uint32_t dim : shape) {
        count *= dim;
    }
    return count;
}

std::vector<uint8_t> CreateSequentialBytes(size_t size)
{
    std::vector<uint8_t> data(size);
    for (size_t i = 0; i < size; ++i) {
        data[i] = static_cast<uint8_t>(i % 251);
    }
    return data;
}

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

void TestHwcTensorToRgbSuccess(int deviceId, ImageProcessor &imageProcessor)
{
    cv::Mat convertedMat = LoadRgbImage();
    std::vector<uint32_t> hwcTensorShape = {
        static_cast<uint32_t>(convertedMat.rows),
        static_cast<uint32_t>(convertedMat.cols),
        static_cast<uint32_t>(convertedMat.channels())
    };

    MxBase::Tensor inputTensor(convertedMat.data, hwcTensorShape, MxBase::TensorDType::UINT8);
    inputTensor.ToDevice(deviceId);
    (void)TensorToImage(inputTensor, ImageFormat::RGB_888);
}

void TestNhwcTensorToRgbSuccess(int deviceId, ImageProcessor &imageProcessor)
{
    cv::Mat convertedMat = LoadRgbImage();
    std::vector<uint32_t> nhwcTensorShape = {
        1,
        static_cast<uint32_t>(convertedMat.rows),
        static_cast<uint32_t>(convertedMat.cols),
        static_cast<uint32_t>(convertedMat.channels())
    };

    MxBase::Tensor inputTensor(convertedMat.data, nhwcTensorShape, MxBase::TensorDType::UINT8);
    inputTensor.ToDevice(deviceId);
    (void)TensorToImage(inputTensor, ImageFormat::RGB_888);
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

void TestWithStrideTensorToYuvSuccess(int deviceId, ImageProcessor &imageProcessor)
{
    std::vector<uint32_t> tensorShape = {17, 15};
    std::vector<uint8_t> tensorData = CreateSequentialBytes(GetElementCount(tensorShape));
    MxBase::Tensor inputTensor(tensorData.data(), tensorShape, TensorDType::UINT8);
    inputTensor.ToDevice(deviceId);
    (void)TensorToImage(inputTensor, ImageFormat::YUV_400);
}

void TestNoStrideTensorToYuvSuccess(int deviceId, ImageProcessor &imageProcessor)
{
    std::vector<uint32_t> tensorShape = {20, 16};
    std::vector<uint8_t> tensorData = CreateSequentialBytes(GetElementCount(tensorShape));
    MxBase::Tensor inputTensor(tensorData.data(), tensorShape, TensorDType::UINT8);
    inputTensor.ToDevice(deviceId);
    (void)TensorToImage(inputTensor, ImageFormat::YUV_400);
}

void TestAnyHeightTensorToYuvSuccess(int deviceId, ImageProcessor &imageProcessor)
{
    std::vector<uint32_t> tensorShape = {16, 2};
    std::vector<uint8_t> tensorData = CreateSequentialBytes(GetElementCount(tensorShape));
    MxBase::Tensor inputTensor(tensorData.data(), tensorShape, TensorDType::UINT8);
    inputTensor.ToDevice(deviceId);
    (void)TensorToImage(inputTensor, ImageFormat::YUV_400);
}

void TestAnyHeightTensorToRgbSuccess(int deviceId, ImageProcessor &imageProcessor)
{
    std::vector<uint32_t> tensorShape = {480, 480, 3};
    std::vector<uint8_t> tensorData = CreateSequentialBytes(GetElementCount(tensorShape));
    MxBase::Tensor inputTensor(tensorData.data(), tensorShape, TensorDType::UINT8);
    inputTensor.ToDevice(deviceId);
    (void)TensorToImage(inputTensor, ImageFormat::RGB_888);
}

void TestUint32TensorToRgbFailed(int deviceId, ImageProcessor &imageProcessor)
{
    std::vector<uint32_t> threeDimTensorShape = {3, 4, 1};
    uint32_t tensorData[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    MxBase::Tensor inputTensor(&tensorData, threeDimTensorShape, TensorDType::UINT32);
    inputTensor.ToDevice(deviceId);
    (void)TensorToImage(inputTensor, ImageFormat::RGB_888);
}

void Test2DimTensorToBgrFailed(int deviceId, ImageProcessor &imageProcessor)
{
    std::vector<uint32_t> twoDimTensorShape = {2, 2};
    uint8_t tensorData[] = {255, 2, 255, 4};
    MxBase::Tensor inputTensor(&tensorData, twoDimTensorShape, TensorDType::UINT8);
    inputTensor.ToDevice(deviceId);
    (void)TensorToImage(inputTensor, ImageFormat::BGR_888);
}

void TestEmptyTensorFailed(int deviceId, ImageProcessor &imageProcessor)
{
    MxBase::Tensor inputTensor;
    (void)TensorToImage(inputTensor, ImageFormat::YUV_400);
}

void TestThreeDimTensorToYuvFailed(int deviceId, ImageProcessor &imageProcessor)
{
    std::vector<uint32_t> tensorShape = {3, 4, 1};
    uint8_t tensorData[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    MxBase::Tensor inputTensor(tensorData, tensorShape, TensorDType::UINT8);
    MxBase::Rect validRoi = {1, 1, 3, 3};
    inputTensor.SetValidRoi(validRoi);
    inputTensor.ToDevice(deviceId);
    (void)TensorToImage(inputTensor, ImageFormat::YUV_400);
}

void TestOneDimTensorToYuvsp422Failed(int deviceId, ImageProcessor &imageProcessor)
{
    std::vector<uint32_t> tensorShape = {12};
    uint8_t tensorData[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    MxBase::Tensor inputTensor(tensorData, tensorShape, TensorDType::UINT8);
    inputTensor.ToDevice(deviceId);
    (void)TensorToImage(inputTensor, ImageFormat::YVU_SP_422);
}

void TestTwoBatchTensorToYuvspFailed(int deviceId, ImageProcessor &imageProcessor)
{
    std::vector<uint32_t> tensorShape = {2, 3, 2, 1};
    uint8_t tensorData[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    MxBase::Tensor inputTensor(tensorData, tensorShape, TensorDType::UINT8);
    inputTensor.ToDevice(deviceId);
    (void)TensorToImage(inputTensor, ImageFormat::YVU_SP_420);
}

}  // namespace

int main(int argc, char *argv[])
{
    if (argc != PARA_NUM) {
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
        ImageProcessor imageProcessor(DEVICE_ID);

        if (functionName == "TestTensorToImageFromImageSuccess") {
            TestTensorToImageFromImageSuccess(DEVICE_ID, imageProcessor);
        } else if (functionName == "TestHwcTensorToRgbSuccess") {
            TestHwcTensorToRgbSuccess(DEVICE_ID, imageProcessor);
        } else if (functionName == "TestNhwcTensorToRgbSuccess") {
            TestNhwcTensorToRgbSuccess(DEVICE_ID, imageProcessor);
        } else if (functionName == "Test2DimTensorToYuvSuccess") {
            Test2DimTensorToYuvSuccess(DEVICE_ID, imageProcessor);
        } else if (functionName == "TestWithStrideTensorToYuvSuccess") {
            TestWithStrideTensorToYuvSuccess(DEVICE_ID, imageProcessor);
        } else if (functionName == "TestNoStrideTensorToYuvSuccess") {
            TestNoStrideTensorToYuvSuccess(DEVICE_ID, imageProcessor);
        } else if (functionName == "TestAnyHeightTensorToYuvSuccess") {
            TestAnyHeightTensorToYuvSuccess(DEVICE_ID, imageProcessor);
        } else if (functionName == "TestAnyHeightTensorToRgbSuccess") {
            TestAnyHeightTensorToRgbSuccess(DEVICE_ID, imageProcessor);
        } else if (functionName == "TestUint32TensorToRgbFailed") {
            TestUint32TensorToRgbFailed(DEVICE_ID, imageProcessor);
        } else if (functionName == "Test2DimTensorToBgrFailed") {
            Test2DimTensorToBgrFailed(DEVICE_ID, imageProcessor);
        } else if (functionName == "TestEmptyTensorFailed") {
            TestEmptyTensorFailed(DEVICE_ID, imageProcessor);
        } else if (functionName == "TestThreeDimTensorToYuvFailed") {
            TestThreeDimTensorToYuvFailed(DEVICE_ID, imageProcessor);
        } else if (functionName == "TestOneDimTensorToYuvsp422Failed") {
            TestOneDimTensorToYuvsp422Failed(DEVICE_ID, imageProcessor);
        } else if (functionName == "TestTwoBatchTensorToYuvspFailed") {
            TestTwoBatchTensorToYuvspFailed(DEVICE_ID, imageProcessor);
        } else {
            std::cout << "Error: Unknown function name: " << functionName << std::endl;
            exit(1);
        }
    }
    MxBase::MxDeInit();
    return 0;
}
