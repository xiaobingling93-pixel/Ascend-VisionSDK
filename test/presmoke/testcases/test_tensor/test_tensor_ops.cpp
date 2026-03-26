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
#include <vector>
#include "MxBase/E2eInfer/ImageProcessor/ImageProcessor.h"
#include "MxBase/MxBase.h"
#include "MxBase/DeviceManager/DeviceManager.h"
#include "MxBase/Asynchron/AscendStream.h"
#include "MxBase/E2eInfer/Tensor/Tensor.h"

using namespace std;
using namespace MxBase;

namespace {

constexpr int PARA_NUM = 2;
constexpr int DEVICE_ID = 0;
constexpr int OTHER_DEVICE_ID = 1;
constexpr float DEFAULT_SCALE = 2.0f;
constexpr float DEFAULT_BIAS = 3.0f;

[[noreturn]] void ExitCaseFailure()
{
    MxDeInit();
    exit(1);
}

[[noreturn]] void ExitCaseFailure(MxBase::AscendStream &stream)
{
    stream.DestroyAscendStream();
    MxDeInit();
    exit(1);
}

uint16_t FloatToHalf(float input)
{
    uint32_t rawValue = *reinterpret_cast<uint32_t *>(&input);
    uint32_t nonSignBits = rawValue & 0x7fffffffu;
    uint32_t signBit = rawValue & 0x80000000u;
    uint32_t exponent = rawValue & 0x7f800000u;

    nonSignBits >>= 13u;
    signBit >>= 16u;
    nonSignBits -= 0x1c000u;
    nonSignBits = (exponent < 0x38800000u) ? 0u : nonSignBits;
    nonSignBits = (exponent > 0x8e000000u) ? 0x7bffu : nonSignBits;
    nonSignBits = (exponent == 0u) ? 0u : nonSignBits;
    nonSignBits |= signBit;

    return static_cast<uint16_t>(nonSignBits);
}

void FillFloat16Buffer(uint16_t *buffer, size_t count, float value)
{
    uint16_t halfValue = FloatToHalf(value);
    for (size_t i = 0; i < count; ++i) {
        buffer[i] = halfValue;
    }
}

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

void TestRescaleIntCoefficientSyncSuccess(int deviceId)
{
    uint8_t input[2][5][1] = {9, 8, 7, 6, 5, 4, 3, 2, 1, 0};
    std::vector<uint32_t> shape = {2, 5, 1};

    MxBase::Tensor inputTensor(input, shape, TensorDType::UINT8);
    inputTensor.ToDevice(deviceId);

    MxBase::Tensor outputTensor;
    int alpha = 1;
    int beta = 2;
    APP_ERROR ret = Rescale(inputTensor, outputTensor, alpha, beta);
    if (ret != APP_ERR_OK) {
        ExitCaseFailure();
    }
}

void TestRescaleLargeBiasSyncSuccess(int deviceId)
{
    uint8_t input[2][5][1] = {9, 8, 7, 6, 5, 4, 3, 2, 1, 0};
    std::vector<uint32_t> shape = {2, 5, 1};

    MxBase::Tensor inputTensor(input, shape, TensorDType::UINT8);
    inputTensor.ToDevice(deviceId);

    MxBase::Tensor outputTensor;
    APP_ERROR ret = Rescale(inputTensor, outputTensor, DEFAULT_SCALE, 255.0f);
    if (ret != APP_ERR_OK) {
        ExitCaseFailure();
    }
}

void TestRescaleZeroScaleSyncSuccess(int deviceId)
{
    uint8_t input[2][5][1] = {9, 8, 7, 6, 5, 4, 3, 2, 1, 0};
    std::vector<uint32_t> shape = {2, 5, 1};

    MxBase::Tensor inputTensor(input, shape, TensorDType::UINT8);
    inputTensor.ToDevice(deviceId);

    MxBase::Tensor outputTensor;
    APP_ERROR ret = Rescale(inputTensor, outputTensor, 0.0f, -1.0f);
    if (ret != APP_ERR_OK) {
        ExitCaseFailure();
    }
}

void TestRescaleFloat32VectorSyncSuccess(int deviceId)
{
    float input[10] = {9, 8, 7, 6, 5, 4, 3, 2, 1, 0};
    std::vector<uint32_t> shape = {10};

    MxBase::Tensor inputTensor(input, shape, TensorDType::FLOAT32);
    inputTensor.ToDevice(deviceId);

    MxBase::Tensor outputTensor;
    APP_ERROR ret = Rescale(inputTensor, outputTensor, DEFAULT_SCALE, DEFAULT_BIAS);
    if (ret != APP_ERR_OK) {
        ExitCaseFailure();
    }
}

void TestRescaleUint8PreallocatedSyncSuccess(int deviceId)
{
    uint8_t input[2][5][1] = {9, 8, 7, 6, 5, 4, 3, 2, 1, 0};
    std::vector<uint32_t> shape = {2, 5, 1};

    MxBase::Tensor inputTensor(input, shape, TensorDType::UINT8);
    inputTensor.ToDevice(deviceId);

    MxBase::Tensor outputTensor(shape, TensorDType::UINT8);
    outputTensor.Malloc();
    outputTensor.ToDevice(deviceId);
    APP_ERROR ret = Rescale(inputTensor, outputTensor, DEFAULT_SCALE, DEFAULT_BIAS);
    if (ret != APP_ERR_OK) {
        ExitCaseFailure();
    }
}

void TestRescaleUint8NoMallocOutputSyncSuccess(int deviceId)
{
    uint8_t input[2][5][1] = {9, 8, 7, 6, 5, 4, 3, 2, 1, 0};
    std::vector<uint32_t> inputShape = {2, 5, 1};
    std::vector<uint32_t> outputShape = {15, 15, 1};

    MxBase::Tensor inputTensor(input, inputShape, TensorDType::UINT8);
    inputTensor.ToDevice(deviceId);

    MxBase::Tensor outputTensor(outputShape, TensorDType::UINT8);
    APP_ERROR ret = Rescale(inputTensor, outputTensor, DEFAULT_SCALE, DEFAULT_BIAS);
    if (ret != APP_ERR_OK) {
        ExitCaseFailure();
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

void TestRescaleUint8FiveDimSyncFailed(int deviceId)
{
    uint8_t input[1][2][5][1][1] = {9, 8, 7, 6, 5, 4, 3, 2, 1, 0};
    std::vector<uint32_t> shape = {1, 2, 5, 1, 1};

    MxBase::Tensor inputTensor(input, shape, TensorDType::UINT8);
    inputTensor.ToDevice(deviceId);

    MxBase::Tensor outputTensor;
    APP_ERROR ret = Rescale(inputTensor, outputTensor, DEFAULT_SCALE, DEFAULT_BIAS);
    if (ret != APP_ERR_OK) {
        ExitCaseFailure();
    }
}

void TestRescaleUint8Float16OutputSyncFailed(int deviceId)
{
    uint8_t input[2][5][1] = {9, 8, 7, 6, 5, 4, 3, 2, 1, 0};
    std::vector<uint32_t> shape = {2, 5, 1};

    MxBase::Tensor inputTensor(input, shape, TensorDType::UINT8);
    inputTensor.ToDevice(deviceId);

    MxBase::Tensor outputTensor(shape, TensorDType::FLOAT16);
    outputTensor.Malloc();
    outputTensor.ToDevice(deviceId);
    APP_ERROR ret = Rescale(inputTensor, outputTensor, DEFAULT_SCALE, DEFAULT_BIAS);
    if (ret != APP_ERR_OK) {
        ExitCaseFailure();
    }
}

void TestRescaleFloat16AsyncSuccess(int deviceId)
{
    uint16_t input[2][5];
    FillFloat16Buffer(&input[0][0], 10, 10.0f);
    std::vector<uint32_t> shape = {2, 5};

    MxBase::Tensor inputTensor(input, shape, TensorDType::FLOAT16);
    inputTensor.ToDevice(deviceId);

    MxBase::Tensor outputTensor;
    MxBase::AscendStream stream(deviceId);
    stream.CreateAscendStream();
    APP_ERROR ret = Rescale(inputTensor, outputTensor, DEFAULT_SCALE, DEFAULT_BIAS, stream);
    if (ret != APP_ERR_OK) {
        ExitCaseFailure(stream);
    }
    stream.Synchronize();
    stream.DestroyAscendStream();
}

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

void TestRescaleStreamDeviceMismatchAsyncFailed(int deviceId)
{
    uint8_t input[2][5][1] = {9, 8, 7, 6, 5, 4, 3, 2, 1, 0};
    std::vector<uint32_t> shape = {2, 5, 1};

    MxBase::Tensor inputTensor(input, shape, TensorDType::UINT8);
    APP_ERROR moveRet = inputTensor.ToDevice(OTHER_DEVICE_ID);
    if (moveRet != APP_ERR_OK) {
        ExitCaseFailure();
    }

    MxBase::Tensor outputTensor;
    MxBase::AscendStream stream(deviceId);
    stream.CreateAscendStream();
    APP_ERROR ret = Rescale(inputTensor, outputTensor, DEFAULT_SCALE, DEFAULT_BIAS, stream);
    if (ret != APP_ERR_OK) {
        ExitCaseFailure(stream);
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

}  // namespace

int main(int argc, char *argv[])
{
    if (argc != PARA_NUM) {
        std::cout << "There should be two parameters." << std::endl;
        return 1;
    }

    string functionName = argv[1];
    APP_ERROR ret = MxInit();
    if (ret != APP_ERR_OK) {
        std::cout << "MxInit failed" << std::endl;
        return 1;
    }
    {
        if (functionName == "TestRescaleUint8SyncSuccess") {
            TestRescaleUint8SyncSuccess(DEVICE_ID);
        } else if (functionName == "TestRescaleIntCoefficientSyncSuccess") {
            TestRescaleIntCoefficientSyncSuccess(DEVICE_ID);
        } else if (functionName == "TestRescaleLargeBiasSyncSuccess") {
            TestRescaleLargeBiasSyncSuccess(DEVICE_ID);
        } else if (functionName == "TestRescaleZeroScaleSyncSuccess") {
            TestRescaleZeroScaleSyncSuccess(DEVICE_ID);
        } else if (functionName == "TestRescaleFloat32VectorSyncSuccess") {
            TestRescaleFloat32VectorSyncSuccess(DEVICE_ID);
        } else if (functionName == "TestRescaleUint8PreallocatedSyncSuccess") {
            TestRescaleUint8PreallocatedSyncSuccess(DEVICE_ID);
        } else if (functionName == "TestRescaleUint8NoMallocOutputSyncSuccess") {
            TestRescaleUint8NoMallocOutputSyncSuccess(DEVICE_ID);
        } else if (functionName == "TestRescaleInt32SyncFailed") {
            TestRescaleInt32SyncFailed(DEVICE_ID);
        } else if (functionName == "TestRescaleUint8FiveDimSyncFailed") {
            TestRescaleUint8FiveDimSyncFailed(DEVICE_ID);
        } else if (functionName == "TestRescaleUint8Float16OutputSyncFailed") {
            TestRescaleUint8Float16OutputSyncFailed(DEVICE_ID);
        } else if (functionName == "TestRescaleFloat16AsyncSuccess") {
            TestRescaleFloat16AsyncSuccess(DEVICE_ID);
        } else if (functionName == "TestRescaleFloat32AsyncSuccess") {
            TestRescaleFloat32AsyncSuccess(DEVICE_ID);
        } else if (functionName == "TestRescaleStreamDeviceMismatchAsyncFailed") {
            TestRescaleStreamDeviceMismatchAsyncFailed(DEVICE_ID);
        } else if (functionName == "TestRescaleInt32AsyncFailed") {
            TestRescaleInt32AsyncFailed(DEVICE_ID);
        } else {
            std::cout << "Unknown function: " << functionName << std::endl;
            exit(1);
        }
    }
    MxDeInit();
    std::cout << "All processes completed" << std::endl;
    return 0;
}
