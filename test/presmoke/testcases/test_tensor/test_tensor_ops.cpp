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

#include "MxBase/E2eInfer/ImageProcessor/ImageProcessor.h"
#include "MxBase/MxBase.h"
#include "MxBase/DeviceManager/DeviceManager.h"
#include "MxBase/Asynchron/AscendStream.h"
#include "MxBase/E2eInfer/Tensor/Tensor.h"

using namespace std;
using namespace MxBase;

// Constants
constexpr float DEFAULT_SCALE = 2.0f;
constexpr float DEFAULT_BIAS = 3.0f;

// Synchronous test cases
void TestRescaleUint8SyncSuccess(int deviceId)
{
    uint8_t input[1][2][5][1] = {9, 8, 7, 6, 5, 4, 3, 2, 1, 0};
    std::vector<uint32_t> shape = {1, 2, 5, 1};

    MxBase::Tensor inputTensor(&input, shape, TensorDType::UINT8); // UINT8
    inputTensor.ToDevice(deviceId);

    MxBase::Tensor outputTensor;
    APP_ERROR ret = Rescale(inputTensor, outputTensor, DEFAULT_SCALE, DEFAULT_BIAS);
    if (ret != APP_ERR_OK) {
        exit(1);
    }
}

void TestRescaleInt32SyncFailed(int deviceId)
{
    int32_t input[2][5][1] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    std::vector<uint32_t> shape = {2, 5, 1};

    MxBase::Tensor inputTensor(&input, shape, TensorDType::INT32); // INT32
    inputTensor.ToDevice(deviceId);

    MxBase::Tensor outputTensor;
    APP_ERROR ret = Rescale(inputTensor, outputTensor, DEFAULT_SCALE, DEFAULT_BIAS);
    if (ret != APP_ERR_OK) {
        exit(1);
    }
}

// Asynchronous test cases
void TestRescaleFloat32AsyncSuccess(int deviceId)
{
    float input[1][2][5][1] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    std::vector<uint32_t> shape = {1, 2, 5, 1};

    MxBase::Tensor inputTensor(&input, shape, TensorDType::FLOAT32); // FLOAT32
    inputTensor.ToDvpp(deviceId);

    MxBase::Tensor outputTensor(shape, TensorDType::FLOAT32);
    outputTensor.Malloc();
    outputTensor.ToDevice(deviceId);
    MxBase::AscendStream stream(deviceId);
    stream.CreateAscendStream();
    APP_ERROR ret = Rescale(inputTensor, outputTensor, DEFAULT_SCALE, DEFAULT_BIAS, stream);
    if (ret != APP_ERR_OK) {
        stream.DestroyAscendStream();
        exit(1);
    }
    stream.Synchronize();
    stream.DestroyAscendStream();
}

void TestRescaleInt32AsyncFailed(int deviceId)
{
    int32_t input[2][5][1] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    std::vector<uint32_t> shape = {2, 5, 1};

    MxBase::Tensor inputTensor(&input, shape, TensorDType::INT32); // INT32
    inputTensor.ToDevice(deviceId);
    MxBase::Tensor outputTensor;
    MxBase::AscendStream stream(deviceId);
    stream.CreateAscendStream();
    APP_ERROR ret = Rescale(inputTensor, outputTensor, DEFAULT_SCALE, DEFAULT_BIAS, stream);
    if (ret != APP_ERR_OK) {
        stream.DestroyAscendStream();
        exit(1);
    }
    stream.Synchronize();
    stream.DestroyAscendStream();
}

int main(int argc, char *argv[])
{
    if (argc != 2) {
        std::cout << "There should be two parameters." << std::endl;
        return 1;
    }

    string functionName = argv[1];
    constexpr int deviceId = 0;
    APP_ERROR ret = MxInit();
    if (ret != APP_ERR_OK) {
        std::cout << "MxInit failed" << std::endl;
        return 1;
    }
    {
        if (functionName == "TestRescaleUint8SyncSuccess") {
            TestRescaleUint8SyncSuccess(deviceId);
        } else if (functionName == "TestRescaleInt32SyncFailed") {
            TestRescaleInt32SyncFailed(deviceId);
        } else if (functionName == "TestRescaleFloat32AsyncSuccess") {
            TestRescaleFloat32AsyncSuccess(deviceId);
        } else if (functionName == "TestRescaleInt32AsyncFailed") {
            TestRescaleInt32AsyncFailed(deviceId);
        } else {
            std::cout << "Unknown function: " << functionName << std::endl;
            exit(1);
        }
    }
    MxDeInit();
    std::cout << "All processes completed" << std::endl;
    return 0;
}