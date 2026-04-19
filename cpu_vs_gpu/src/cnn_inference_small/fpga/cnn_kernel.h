#ifndef CNN_KERNEL_H
#define CNN_KERNEL_H

// Network parameters
constexpr int INPUT_SIZE = 128;
constexpr int CONV1_FILTERS = 32;
constexpr int CONV1_KERNEL = 3;
constexpr int CONV2_FILTERS = 64;
constexpr int CONV2_KERNEL = 3;
constexpr int DENSE1_SIZE = 128;
constexpr int OUTPUT_SIZE = 10;

// Computed sizes
constexpr int CONV1_OUTPUT_SIZE = INPUT_SIZE - CONV1_KERNEL + 1;  // 126
constexpr int CONV2_OUTPUT_SIZE = CONV1_OUTPUT_SIZE - CONV2_KERNEL + 1;  // 124
constexpr int DENSE1_INPUT_SIZE = CONV2_FILTERS * CONV2_OUTPUT_SIZE;  // 7936

// Main FP32 kernel
void cnn_inference_kernel(
    const float input[INPUT_SIZE],
    const float conv1_weights[CONV1_FILTERS * CONV1_KERNEL],
    const float conv1_bias[CONV1_FILTERS],
    const float conv2_weights[CONV2_FILTERS * CONV1_FILTERS * CONV2_KERNEL],
    const float conv2_bias[CONV2_FILTERS],
    const float dense1_weights[DENSE1_SIZE * DENSE1_INPUT_SIZE],
    const float dense1_bias[DENSE1_SIZE],
    const float dense2_weights[OUTPUT_SIZE * DENSE1_SIZE],
    const float dense2_bias[OUTPUT_SIZE],
    float output[OUTPUT_SIZE]
);

// INT8 quantized kernel (for higher throughput)
typedef char int8_t;

void cnn_inference_kernel_int8(
    const int8_t input[INPUT_SIZE],
    const int8_t conv1_weights[CONV1_FILTERS * CONV1_KERNEL],
    const int8_t conv1_bias[CONV1_FILTERS],
    const int8_t conv2_weights[CONV2_FILTERS * CONV1_FILTERS * CONV2_KERNEL],
    const int8_t conv2_bias[CONV2_FILTERS],
    const int8_t dense1_weights[DENSE1_SIZE * DENSE1_INPUT_SIZE], 
    const int8_t dense1_bias[DENSE1_SIZE],
    const int8_t dense2_weights[OUTPUT_SIZE * DENSE1_SIZE],
    const int8_t dense2_bias[OUTPUT_SIZE],
    float output[OUTPUT_SIZE],
    float input_scale,
    float conv1_scale,
    float conv2_scale,
    float dense1_scale,
    float dense2_scale
);

#endif // CNN_KERNEL_H
