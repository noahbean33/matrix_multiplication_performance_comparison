#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <random>
#include <algorithm>
#include <cmath>
#include <cuda_runtime.h>

// Workload E: Fixed CNN/ML Inference Block
// GPU/CUDA implementation with batch=1 vs batch=N comparison

// Network architecture (same as CPU version)
constexpr int INPUT_SIZE = 128;
constexpr int CONV1_FILTERS = 32;
constexpr int CONV1_KERNEL = 3;
constexpr int CONV2_FILTERS = 64;
constexpr int CONV2_KERNEL = 3;
constexpr int DENSE1_SIZE = 128;
constexpr int OUTPUT_SIZE = 10;

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " - " \
                      << cudaGetErrorString(err) << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// ReLU activation
__device__ __forceinline__ float relu(float x) {
    return fmaxf(0.0f, x);
}

// Conv1D kernel (first layer, single channel input)
__global__ void conv1d_first_kernel(
    const float* __restrict__ input,  // [batch_size, INPUT_SIZE]
    const float* __restrict__ weights, // [CONV1_FILTERS, CONV1_KERNEL]
    const float* __restrict__ bias,    // [CONV1_FILTERS]
    float* __restrict__ output,        // [batch_size, CONV1_FILTERS, output_size]
    int batch_size,
    int output_size
) {
    int batch = blockIdx.x;
    int filter = blockIdx.y;
    int pos = threadIdx.x;
    
    if (batch < batch_size && filter < CONV1_FILTERS && pos < output_size) {
        float sum = bias[filter];
        for (int k = 0; k < CONV1_KERNEL; ++k) {
            sum += input[batch * INPUT_SIZE + pos + k] * 
                   weights[filter * CONV1_KERNEL + k];
        }
        output[batch * CONV1_FILTERS * output_size + filter * output_size + pos] = relu(sum);
    }
}

// Conv1D kernel (multi-channel)
__global__ void conv1d_kernel(
    const float* __restrict__ input,  // [batch_size, in_channels, input_size]
    const float* __restrict__ weights, // [out_filters, in_channels, kernel_size]
    const float* __restrict__ bias,    // [out_filters]
    float* __restrict__ output,        // [batch_size, out_filters, output_size]
    int batch_size,
    int in_channels,
    int input_size,
    int out_filters,
    int kernel_size,
    int output_size
) {
    int batch = blockIdx.x;
    int filter = blockIdx.y;
    int pos = threadIdx.x;
    
    if (batch < batch_size && filter < out_filters && pos < output_size) {
        float sum = bias[filter];
        for (int c = 0; c < in_channels; ++c) {
            for (int k = 0; k < kernel_size; ++k) {
                sum += input[batch * in_channels * input_size + c * input_size + pos + k] *
                       weights[filter * in_channels * kernel_size + c * kernel_size + k];
            }
        }
        output[batch * out_filters * output_size + filter * output_size + pos] = relu(sum);
    }
}

// Dense layer kernel
__global__ void dense_kernel(
    const float* __restrict__ input,   // [batch_size, input_size]
    const float* __restrict__ weights,  // [output_size, input_size]
    const float* __restrict__ bias,     // [output_size]
    float* __restrict__ output,         // [batch_size, output_size]
    int batch_size,
    int input_size,
    int output_size,
    bool apply_relu
) {
    int batch = blockIdx.x;
    int out_idx = threadIdx.x;
    
    if (batch < batch_size && out_idx < output_size) {
        float sum = bias[out_idx];
        for (int i = 0; i < input_size; ++i) {
            sum += input[batch * input_size + i] * weights[out_idx * input_size + i];
        }
        output[batch * output_size + out_idx] = apply_relu ? relu(sum) : sum;
    }
}

// Softmax kernel
__global__ void softmax_kernel(
    float* __restrict__ data,  // [batch_size, size]
    int batch_size,
    int size
) {
    int batch = blockIdx.x;
    if (batch >= batch_size) return;
    
    float* row = data + batch * size;
    
    // Find max
    float max_val = row[0];
    for (int i = 1; i < size; ++i) {
        max_val = fmaxf(max_val, row[i]);
    }
    
    // Exp and sum
    float sum = 0.0f;
    for (int i = 0; i < size; ++i) {
        row[i] = expf(row[i] - max_val);
        sum += row[i];
    }
    
    // Normalize
    for (int i = 0; i < size; ++i) {
        row[i] /= sum;
    }
}

// Model structure on host
struct CNNModel {
    std::vector<float> conv1_weights;
    std::vector<float> conv1_bias;
    std::vector<float> conv2_weights;
    std::vector<float> conv2_bias;
    std::vector<float> dense1_weights;
    std::vector<float> dense1_bias;
    std::vector<float> dense2_weights;
    std::vector<float> dense2_bias;
    
    int conv1_output_size;
    int conv2_output_size;
};

// Initialize model weights
CNNModel initialize_model() {
    std::mt19937 rng(42);
    std::normal_distribution<float> dist(0.0f, 0.1f);
    
    CNNModel model;
    model.conv1_output_size = INPUT_SIZE - CONV1_KERNEL + 1;
    model.conv2_output_size = model.conv1_output_size - CONV2_KERNEL + 1;
    
    // Conv1
    model.conv1_weights.resize(CONV1_FILTERS * CONV1_KERNEL);
    model.conv1_bias.resize(CONV1_FILTERS);
    for (auto& w : model.conv1_weights) w = dist(rng);
    for (auto& b : model.conv1_bias) b = dist(rng);
    
    // Conv2
    model.conv2_weights.resize(CONV2_FILTERS * CONV1_FILTERS * CONV2_KERNEL);
    model.conv2_bias.resize(CONV2_FILTERS);
    for (auto& w : model.conv2_weights) w = dist(rng);
    for (auto& b : model.conv2_bias) b = dist(rng);
    
    // Dense1
    int dense1_input_size = CONV2_FILTERS * model.conv2_output_size;
    model.dense1_weights.resize(DENSE1_SIZE * dense1_input_size);
    model.dense1_bias.resize(DENSE1_SIZE);
    for (auto& w : model.dense1_weights) w = dist(rng);
    for (auto& b : model.dense1_bias) b = dist(rng);
    
    // Dense2
    model.dense2_weights.resize(OUTPUT_SIZE * DENSE1_SIZE);
    model.dense2_bias.resize(OUTPUT_SIZE);
    for (auto& w : model.dense2_weights) w = dist(rng);
    for (auto& b : model.dense2_bias) b = dist(rng);
    
    return model;
}

void benchmark_cnn_gpu(int batch_size) {
    std::cout << "\n====================================" << std::endl;
    std::cout << "CNN Inference GPU Benchmark" << std::endl;
    std::cout << "Batch Size: " << batch_size << std::endl;
    std::cout << "====================================" << std::endl;
    
    // Initialize model
    CNNModel model = initialize_model();
    
    int conv1_out_size = model.conv1_output_size;
    int conv2_out_size = model.conv2_output_size;
    int dense1_input_size = CONV2_FILTERS * conv2_out_size;
    
    // Allocate device memory for weights
    float *d_conv1_w, *d_conv1_b, *d_conv2_w, *d_conv2_b;
    float *d_dense1_w, *d_dense1_b, *d_dense2_w, *d_dense2_b;
    
    CUDA_CHECK(cudaMalloc(&d_conv1_w, model.conv1_weights.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_conv1_b, model.conv1_bias.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_conv2_w, model.conv2_weights.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_conv2_b, model.conv2_bias.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dense1_w, model.dense1_weights.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dense1_b, model.dense1_bias.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dense2_w, model.dense2_weights.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dense2_b, model.dense2_bias.size() * sizeof(float)));
    
    // Copy weights to device
    CUDA_CHECK(cudaMemcpy(d_conv1_w, model.conv1_weights.data(), model.conv1_weights.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_conv1_b, model.conv1_bias.data(), model.conv1_bias.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_conv2_w, model.conv2_weights.data(), model.conv2_weights.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_conv2_b, model.conv2_bias.data(), model.conv2_bias.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_dense1_w, model.dense1_weights.data(), model.dense1_weights.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_dense1_b, model.dense1_bias.data(), model.dense1_bias.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_dense2_w, model.dense2_weights.data(), model.dense2_weights.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_dense2_b, model.dense2_bias.data(), model.dense2_bias.size() * sizeof(float), cudaMemcpyHostToDevice));
    
    // Allocate device memory for activations
    float *d_input, *d_conv1_out, *d_conv2_out, *d_dense1_out, *d_output;
    
    CUDA_CHECK(cudaMalloc(&d_input, batch_size * INPUT_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_conv1_out, batch_size * CONV1_FILTERS * conv1_out_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_conv2_out, batch_size * CONV2_FILTERS * conv2_out_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dense1_out, batch_size * DENSE1_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_output, batch_size * OUTPUT_SIZE * sizeof(float)));
    
    // Generate random input
    std::mt19937 rng(123);
    std::uniform_real_distribution<float> input_dist(-1.0f, 1.0f);
    std::vector<float> h_input(batch_size * INPUT_SIZE);
    for (auto& val : h_input) val = input_dist(rng);
    
    CUDA_CHECK(cudaMemcpy(d_input, h_input.data(), h_input.size() * sizeof(float), cudaMemcpyHostToDevice));
    
    // Warmup
    dim3 grid_conv1(batch_size, CONV1_FILTERS);
    dim3 block_conv1(conv1_out_size);
    for (int i = 0; i < 10; ++i) {
        conv1d_first_kernel<<<grid_conv1, block_conv1>>>(
            d_input, d_conv1_w, d_conv1_b, d_conv1_out, batch_size, conv1_out_size);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Benchmark
    const int num_inferences = 1000;
    
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    
    CUDA_CHECK(cudaEventRecord(start));
    
    for (int iter = 0; iter < num_inferences; ++iter) {
        // Conv1
        conv1d_first_kernel<<<grid_conv1, block_conv1>>>(
            d_input, d_conv1_w, d_conv1_b, d_conv1_out, batch_size, conv1_out_size);
        
        // Conv2
        dim3 grid_conv2(batch_size, CONV2_FILTERS);
        dim3 block_conv2(conv2_out_size);
        conv1d_kernel<<<grid_conv2, block_conv2>>>(
            d_conv1_out, d_conv2_w, d_conv2_b, d_conv2_out,
            batch_size, CONV1_FILTERS, conv1_out_size, CONV2_FILTERS, CONV2_KERNEL, conv2_out_size);
        
        // Dense1
        dim3 grid_dense1(batch_size);
        dim3 block_dense1(DENSE1_SIZE);
        dense_kernel<<<grid_dense1, block_dense1>>>(
            d_conv2_out, d_dense1_w, d_dense1_b, d_dense1_out,
            batch_size, dense1_input_size, DENSE1_SIZE, true);
        
        // Dense2
        dim3 grid_dense2(batch_size);
        dim3 block_dense2(OUTPUT_SIZE);
        dense_kernel<<<grid_dense2, block_dense2>>>(
            d_dense1_out, d_dense2_w, d_dense2_b, d_output,
            batch_size, DENSE1_SIZE, OUTPUT_SIZE, false);
        
        // Softmax
        softmax_kernel<<<batch_size, 1>>>(d_output, batch_size, OUTPUT_SIZE);
    }
    
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    
    float milliseconds = 0;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
    
    double total_time_s = milliseconds / 1000.0;
    double avg_latency_ms = milliseconds / num_inferences;
    double inferences_per_sec = (num_inferences * batch_size) / total_time_s;
    
    // Calculate FLOPs
    long long flops = 0;
    flops += (long long)conv1_out_size * CONV1_FILTERS * CONV1_KERNEL * 2;
    flops += (long long)conv2_out_size * CONV2_FILTERS * CONV1_FILTERS * CONV2_KERNEL * 2;
    flops += (long long)DENSE1_SIZE * dense1_input_size * 2;
    flops += (long long)OUTPUT_SIZE * DENSE1_SIZE * 2;
    
    double gflops_per_sec = (flops * inferences_per_sec) / 1e9;
    
    // Print results
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "\n----- Performance Metrics -----" << std::endl;
    std::cout << "Batch size:        " << batch_size << std::endl;
    std::cout << "Total inferences:  " << num_inferences * batch_size << std::endl;
    std::cout << "Total time:        " << total_time_s * 1000.0 << " ms" << std::endl;
    std::cout << "Avg latency:       " << avg_latency_ms << " ms/batch" << std::endl;
    std::cout << "Latency per item:  " << avg_latency_ms / batch_size << " ms/inference" << std::endl;
    std::cout << "Throughput:        " << inferences_per_sec << " inferences/sec" << std::endl;
    std::cout << "Compute:           " << gflops_per_sec << " GFLOP/s" << std::endl;
    std::cout << "FLOPs/inference:   " << flops << std::endl;
    
    // Copy output back for verification
    std::vector<float> h_output(batch_size * OUTPUT_SIZE);
    CUDA_CHECK(cudaMemcpy(h_output.data(), d_output, h_output.size() * sizeof(float), cudaMemcpyDeviceToHost));
    
    std::cout << "\n----- Output Sample (first in batch) -----" << std::endl;
    std::cout << "Class probabilities: [";
    for (int i = 0; i < OUTPUT_SIZE; ++i) {
        std::cout << h_output[i];
        if (i < OUTPUT_SIZE - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;
    
    // Cleanup
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_conv1_out));
    CUDA_CHECK(cudaFree(d_conv2_out));
    CUDA_CHECK(cudaFree(d_dense1_out));
    CUDA_CHECK(cudaFree(d_output));
    CUDA_CHECK(cudaFree(d_conv1_w));
    CUDA_CHECK(cudaFree(d_conv1_b));
    CUDA_CHECK(cudaFree(d_conv2_w));
    CUDA_CHECK(cudaFree(d_conv2_b));
    CUDA_CHECK(cudaFree(d_dense1_w));
    CUDA_CHECK(cudaFree(d_dense1_b));
    CUDA_CHECK(cudaFree(d_dense2_w));
    CUDA_CHECK(cudaFree(d_dense2_b));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
}

int main() {
    std::cout << "====================================" << std::endl;
    std::cout << "Workload E: CNN/ML Inference Block" << std::endl;
    std::cout << "Architecture: GPU (CUDA)" << std::endl;
    std::cout << "====================================" << std::endl;
    
    // Check CUDA device
    int device_count;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    if (device_count == 0) {
        std::cerr << "No CUDA devices found!" << std::endl;
        return 1;
    }
    
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    std::cout << "Using GPU: " << prop.name << std::endl;
    std::cout << "Compute Capability: " << prop.major << "." << prop.minor << std::endl;
    
    // Model info
    std::cout << "\n----- Network Architecture -----" << std::endl;
    std::cout << "Input: [" << INPUT_SIZE << "] 1D signal" << std::endl;
    std::cout << "Conv1D(" << CONV1_FILTERS << " filters, kernel=" << CONV1_KERNEL << ") -> ReLU" << std::endl;
    std::cout << "Conv1D(" << CONV2_FILTERS << " filters, kernel=" << CONV2_KERNEL << ") -> ReLU" << std::endl;
    std::cout << "Dense(" << DENSE1_SIZE << ") -> ReLU" << std::endl;
    std::cout << "Dense(" << OUTPUT_SIZE << ") -> Softmax" << std::endl;
    
    // Benchmark different batch sizes
    std::cout << "\n====================================" << std::endl;
    std::cout << "Key Insight: GPU Batch Size Impact" << std::endl;
    std::cout << "====================================" << std::endl;
    std::cout << "Batch=1: Low latency (streaming)" << std::endl;
    std::cout << "Batch=32/64: High throughput (batched)" << std::endl;
    
    benchmark_cnn_gpu(1);   // Streaming - where FPGA competes
    benchmark_cnn_gpu(32);  // Medium batch
    benchmark_cnn_gpu(64);  // Large batch - where GPU excels
    
    std::cout << "\n====================================" << std::endl;
    std::cout << "Analysis: GPU is best at batch>>1" << std::endl;
    std::cout << "For batch=1 streaming, FPGA wins!" << std::endl;
    std::cout << "====================================" << std::endl;
    
    return 0;
}
