#!/bin/bash
TEST_CASES=(
    "test_tensor_to_image" "TestTensorToImageFromImageSuccess" "0"  # Normal test case
    "test_tensor_to_image" "TestHwcTensorToRgbSuccess" "0"  # Normal test case
    "test_tensor_to_image" "TestNhwcTensorToRgbSuccess" "0"  # Normal test case
    "test_tensor_to_image" "Test2DimTensorToYuvSuccess" "0"        # Normal test case
    "test_tensor_to_image" "TestWithStrideTensorToYuvSuccess" "0"  # Normal test case
    "test_tensor_to_image" "TestNoStrideTensorToYuvSuccess" "0"  # Normal test case
    "test_tensor_to_image" "TestAnyHeightTensorToYuvSuccess" "0"  # Normal test case
    "test_tensor_to_image" "TestAnyHeightTensorToRgbSuccess" "0"  # Normal test case
    "test_tensor_to_image" "TestUint32TensorToRgbFailed" "1"       # Abnormal test case
    "test_tensor_to_image" "Test2DimTensorToBgrFailed" "1"         # Abnormal test case
    "test_tensor_to_image" "TestEmptyTensorFailed" "1"  # Abnormal test case
    "test_tensor_to_image" "TestThreeDimTensorToYuvFailed" "1"  # Abnormal test case
    "test_tensor_to_image" "TestOneDimTensorToYuvsp422Failed" "1"  # Abnormal test case
    "test_tensor_to_image" "TestTwoBatchTensorToYuvspFailed" "1"  # Abnormal test case
    "test_image_to_tensor" "TestYuvsp420ToTensorSuccess" "0"       # Normal test case
    "test_image_to_tensor" "TestYuvsp422ToTensorSuccess" "0"       # Normal test case
)

# Working directory
WORK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="$WORK_DIR/build"
# Executable path will be built in the loop
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source "$WORK_DIR/../../../../presmoke_install/mxVision/set_env.sh"
export LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/driver/lib64/driver:$LD_LIBRARY_PATH
# Delete existing build directory
if [ -d "$BUILD_DIR" ]; then
    rm -rf "$BUILD_DIR"
fi

# Create build directory
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

# Build with cmake
 echo "Building test_tensor_to_image and test_image_to_tensor..."
if ! cmake .. ; then
    echo "CMake configuration failed!"
    exit 1
fi

if ! make ; then
    echo "Build failed!"
    exit 1
fi

echo "Build completed successfully!"

# Statistic test results
PASSED=0
TOTAL=$(( ${#TEST_CASES[@]} / 3 ))

# Run each test case
echo "\nRunning tests..."
echo "===================================="

for ((i=0; i<${#TEST_CASES[@]}; i+=3)); do
    EXECUTABLE_NAME=${TEST_CASES[$i]}
    TEST_FUNC=${TEST_CASES[$i+1]}
    EXPECTED_RET=${TEST_CASES[$i+2]}
    
    # Build executable path
    EXECUTABLE="$WORK_DIR/$EXECUTABLE_NAME"
    
    echo "Running test: ${EXECUTABLE_NAME} ${TEST_FUNC}"
    
    # Run the test
    "$EXECUTABLE" "$TEST_FUNC"
    ACTUAL_RET=$?
    
    # Check result
    if [ "$ACTUAL_RET" -eq "$EXPECTED_RET" ]; then
        echo "PASSED (Expected: $EXPECTED_RET, Actual: $ACTUAL_RET)"
        ((PASSED++))
    else
        echo "FAILED (Expected: $EXPECTED_RET, Actual: $ACTUAL_RET)"
    fi
    
    echo "------------------------------------"
done

# Output test summary
echo "\nTest Summary:"
echo "===================================="
echo "Total tests: $TOTAL"
echo "Passed: $PASSED"
echo "Failed: $((TOTAL - PASSED))"

if [ "$PASSED" -eq "$TOTAL" ]; then
    echo "All tests passed!"
    exit 0
else
    echo "Some tests failed!"
    exit 1
fi
