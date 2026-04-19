#include "cnn_kernel.h"
#include <cmath>

// FPGA CNN Inference Kernel - Optimized for Streaming (batch=1)
// This demonstrates why FPGAs excel at low-latency inference

// Network parameters
constexpr int INPUT_SIZE = 128;
constexpr int CONV1_FILTERS = 32;
constexpr int CONV1_KERNEL = 3;
constexpr int CONV2_FILTERS = 64;
constexpr int CONV2_KERNEL = 3;
constexpr int DENSE1_SIZE = 128;
constexpr int OUTPUT_SIZE = 10;

// ReLU activation
inline float relu(float x) {
#pragma HLS INLINE
    return (x > 0.0f) ? x : 0.0f;
}

// Softmax (simplified for output layer)
void softmax(float output[OUTPUT_SIZE]) {
#pragma HLS INLINE off
    float max_val = output[0];
    FIND_MAX: for (int i = 1; i < OUTPUT_SIZE; ++i) {
#pragma HLS PIPELINE II=1
        if (output[i] > max_val) max_val = output[i];
    }
    
    float sum = 0.0f;
    EXP_SUM: for (int i = 0; i < OUTPUT_SIZE; ++i) {
#pragma HLS PIPELINE II=1
        output[i] = expf(output[i] - max_val);
        sum += output[i];
    }
    
    NORMALIZE: for (int i = 0; i < OUTPUT_SIZE; ++i) {
#pragma HLS PIPELINE II=1
        output[i] /= sum;
    }
}

// Conv1D - First layer (single channel input)
void conv1d_first(
    const float input[INPUT_SIZE],
    const float weights[CONV1_FILTERS][CONV1_KERNEL],
    const float bias[CONV1_FILTERS],
    float output[CONV1_FILTERS * (INPUT_SIZE - CONV1_KERNEL + 1)]
) {
#pragma HLS INLINE off
    
    constexpr int output_size = INPUT_SIZE - CONV1_KERNEL + 1;
    
    // Key FPGA optimization: pipeline the entire computation
    FILTER_LOOP: for (int f = 0; f < CONV1_FILTERS; ++f) {
        POS_LOOP: for (int i = 0; i < output_size; ++i) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_FLATTEN
            
            float sum = bias[f];
            KERNEL_LOOP: for (int k = 0; k < CONV1_KERNEL; ++k) {
                sum += input[i + k] * weights[f][k];
            }
            output[f * output_size + i] = relu(sum);
        }
    }
}

// Conv1D - Multi-channel
void conv1d_multi(
    const float input[], // Flat array
    const float weights[], // [out_filters][in_channels][kernel_size]
    const float bias[],
    float output[],
    int in_channels,
    int input_size,
    int out_filters,
    int kernel_size
) {
#pragma HLS INLINE off
    
    int output_size = input_size - kernel_size + 1;
    
    FILTER: for (int f = 0; f < out_filters; ++f) {
        POS: for (int i = 0; i < output_size; ++i) {
#pragma HLS PIPELINE II=2
            
            float sum = bias[f];
            CHANNEL: for (int c = 0; c < in_channels; ++c) {
                KERNEL: for (int k = 0; k < kernel_size; ++k) {
                    int w_idx = f * in_channels * kernel_size + c * kernel_size + k;
                    int in_idx = c * input_size + i + k;
                    sum += input[in_idx] * weights[w_idx];
                }
            }
            output[f * output_size + i] = relu(sum);
        }
    }
}

// Dense layer with optional ReLU
void dense_layer(
    const float input[],
    const float weights[], // [output_size][input_size]
    const float bias[],
    float output[],
    int input_size,
    int output_size,
    bool apply_relu
) {
#pragma HLS INLINE off
    
    OUTPUT_LOOP: for (int i = 0; i < output_size; ++i) {
        float sum = bias[i];
        
        INPUT_LOOP: for (int j = 0; j < input_size; ++j) {
#pragma HLS PIPELINE II=1
            sum += input[j] * weights[i * input_size + j];
        }
        
        output[i] = apply_relu ? relu(sum) : sum;
    }
}

// Main CNN inference kernel - streaming optimized
void cnn_inference_kernel(
    const float input[INPUT_SIZE],
    const float conv1_weights[CONV1_FILTERS * CONV1_KERNEL],
    const float conv1_bias[CONV1_FILTERS],
    const float conv2_weights[CONV2_FILTERS * CONV1_FILTERS * CONV2_KERNEL],
    const float conv2_bias[CONV2_FILTERS],
    const float dense1_weights[], // Size computed at runtime
    const float dense1_bias[DENSE1_SIZE],
    const float dense2_weights[OUTPUT_SIZE * DENSE1_SIZE],
    const float dense2_bias[OUTPUT_SIZE],
    float output[OUTPUT_SIZE]
) {
#pragma HLS INTERFACE m_axi port=input offset=slave bundle=gmem0 depth=128
#pragma HLS INTERFACE m_axi port=conv1_weights offset=slave bundle=gmem1 depth=96
#pragma HLS INTERFACE m_axi port=conv1_bias offset=slave bundle=gmem1 depth=32
#pragma HLS INTERFACE m_axi port=conv2_weights offset=slave bundle=gmem2 depth=6144
#pragma HLS INTERFACE m_axi port=conv2_bias offset=slave bundle=gmem2 depth=64
#pragma HLS INTERFACE m_axi port=dense1_weights offset=slave bundle=gmem3 depth=1000000
#pragma HLS INTERFACE m_axi port=dense1_bias offset=slave bundle=gmem3 depth=128
#pragma HLS INTERFACE m_axi port=dense2_weights offset=slave bundle=gmem4 depth=1280
#pragma HLS INTERFACE m_axi port=dense2_bias offset=slave bundle=gmem4 depth=10
#pragma HLS INTERFACE m_axi port=output offset=slave bundle=gmem5 depth=10
#pragma HLS INTERFACE s_axilite port=return

    // DATAFLOW optimization: pipeline the entire network
#pragma HLS DATAFLOW
    
    // Intermediate buffers (will be implemented as FIFOs/PIPOs in dataflow)
    constexpr int conv1_out_size = INPUT_SIZE - CONV1_KERNEL + 1; // 126
    constexpr int conv2_out_size = conv1_out_size - CONV2_KERNEL + 1; // 124
    constexpr int dense1_input_size = CONV2_FILTERS * conv2_out_size; // 7936
    
    float conv1_out[CONV1_FILTERS * conv1_out_size];
#pragma HLS BIND_STORAGE variable=conv1_out type=FIFO impl=BRAM
    
    float conv2_out[CONV2_FILTERS * conv2_out_size];
#pragma HLS BIND_STORAGE variable=conv2_out type=FIFO impl=BRAM
    
    float dense1_out[DENSE1_SIZE];
#pragma HLS BIND_STORAGE variable=dense1_out type=FIFO impl=BRAM
    
    float dense2_out[OUTPUT_SIZE];
#pragma HLS ARRAY_PARTITION variable=dense2_out complete
    
    // Conv1: Input -> Conv1
    // Reshape weights for easier access
    float conv1_w_2d[CONV1_FILTERS][CONV1_KERNEL];
    LOAD_CONV1_W: for (int f = 0; f < CONV1_FILTERS; ++f) {
        for (int k = 0; k < CONV1_KERNEL; ++k) {
#pragma HLS PIPELINE II=1
            conv1_w_2d[f][k] = conv1_weights[f * CONV1_KERNEL + k];
        }
    }
    
    conv1d_first(input, conv1_w_2d, conv1_bias, conv1_out);
    
    // Conv2: Conv1 -> Conv2
    conv1d_multi(conv1_out, conv2_weights, conv2_bias, conv2_out,
                 CONV1_FILTERS, conv1_out_size, CONV2_FILTERS, CONV2_KERNEL);
    
    // Dense1: Conv2 -> Dense1
    dense_layer(conv2_out, dense1_weights, dense1_bias, dense1_out,
                dense1_input_size, DENSE1_SIZE, true);
    
    // Dense2: Dense1 -> Output (no ReLU)
    dense_layer(dense1_out, dense2_weights, dense2_bias, dense2_out,
                DENSE1_SIZE, OUTPUT_SIZE, false);
    
    // Softmax
    softmax(dense2_out);
    
    // Copy to output
    COPY_OUT: for (int i = 0; i < OUTPUT_SIZE; ++i) {
#pragma HLS PIPELINE II=1
        output[i] = dense2_out[i];
    }
}

// Variant: INT8 quantized version (for higher throughput)
// This is where FPGAs really shine - custom precision!
typedef char int8_t;

void cnn_inference_kernel_int8(
    const int8_t input[INPUT_SIZE],
    const int8_t conv1_weights[CONV1_FILTERS * CONV1_KERNEL],
    const int8_t conv1_bias[CONV1_FILTERS],
    const int8_t conv2_weights[CONV2_FILTERS * CONV1_FILTERS * CONV2_KERNEL],
    const int8_t conv2_bias[CONV2_FILTERS],
    const int8_t dense1_weights[], 
    const int8_t dense1_bias[DENSE1_SIZE],
    const int8_t dense2_weights[OUTPUT_SIZE * DENSE1_SIZE],
    const int8_t dense2_bias[OUTPUT_SIZE],
    float output[OUTPUT_SIZE],
    float input_scale,
    float conv1_scale,
    float conv2_scale,
    float dense1_scale,
    float dense2_scale
) {
#pragma HLS INTERFACE m_axi port=input offset=slave bundle=gmem0 depth=128
#pragma HLS INTERFACE m_axi port=conv1_weights offset=slave bundle=gmem1 depth=96
#pragma HLS INTERFACE m_axi port=output offset=slave bundle=gmem5 depth=10
#pragma HLS INTERFACE s_axilite port=return
#pragma HLS DATAFLOW

    // INT8 inference with quantization scales
    // This allows 4x memory reduction and 4x+ throughput increase
    // FPGAs can easily implement custom bit-widths (e.g., 6-bit, 4-bit)
    
    // Implementation would use fixed-point arithmetic
    // For demonstration, showing the structure
    
    constexpr int conv1_out_size = INPUT_SIZE - CONV1_KERNEL + 1;
    int conv1_out_int[CONV1_FILTERS * conv1_out_size];
#pragma HLS BIND_STORAGE variable=conv1_out_int type=FIFO impl=URAM
    
    // Quantized MAC operations are much faster on FPGA
    // Can achieve 1 cycle per MAC with DSP slices
    
    // Placeholder for INT8 inference
    // Real implementation would use ap_int<8> and fixed-point math
}
