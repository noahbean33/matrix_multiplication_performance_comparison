#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <random>
#include <cmath>
#include <algorithm>
#include <cstring>

// Workload E: Fixed CNN/ML Inference Block
// CPU implementation - 1D CNN for signal processing

// Network architecture:
// Input: [128] 1D signal
// Conv1D(32 filters, kernel=3) -> ReLU
// Conv1D(64 filters, kernel=3) -> ReLU  
// Flatten
// Dense(128) -> ReLU
// Dense(10) -> Softmax
// Output: [10] class probabilities

constexpr int INPUT_SIZE = 128;
constexpr int CONV1_FILTERS = 32;
constexpr int CONV1_KERNEL = 3;
constexpr int CONV2_FILTERS = 64;
constexpr int CONV2_KERNEL = 3;
constexpr int DENSE1_SIZE = 128;
constexpr int OUTPUT_SIZE = 10;

// Model weights structure
struct CNNModel {
    // Conv1: [CONV1_FILTERS][CONV1_KERNEL]
    std::vector<std::vector<float>> conv1_weights;
    std::vector<float> conv1_bias;
    
    // Conv2: [CONV2_FILTERS][CONV1_FILTERS][CONV2_KERNEL]
    std::vector<std::vector<std::vector<float>>> conv2_weights;
    std::vector<float> conv2_bias;
    
    // Dense1: [DENSE1_SIZE][conv2_output_size]
    std::vector<std::vector<float>> dense1_weights;
    std::vector<float> dense1_bias;
    
    // Dense2: [OUTPUT_SIZE][DENSE1_SIZE]
    std::vector<std::vector<float>> dense2_weights;
    std::vector<float> dense2_bias;
    
    int conv1_output_size;
    int conv2_output_size;
};

// Initialize random weights
CNNModel initialize_model() {
    std::mt19937 rng(42);
    std::normal_distribution<float> dist(0.0f, 0.1f);
    
    CNNModel model;
    
    // Conv1 weights
    model.conv1_weights.resize(CONV1_FILTERS);
    for (int i = 0; i < CONV1_FILTERS; ++i) {
        model.conv1_weights[i].resize(CONV1_KERNEL);
        for (int j = 0; j < CONV1_KERNEL; ++j) {
            model.conv1_weights[i][j] = dist(rng);
        }
    }
    model.conv1_bias.resize(CONV1_FILTERS);
    for (int i = 0; i < CONV1_FILTERS; ++i) {
        model.conv1_bias[i] = dist(rng);
    }
    model.conv1_output_size = INPUT_SIZE - CONV1_KERNEL + 1;
    
    // Conv2 weights
    model.conv2_weights.resize(CONV2_FILTERS);
    for (int i = 0; i < CONV2_FILTERS; ++i) {
        model.conv2_weights[i].resize(CONV1_FILTERS);
        for (int j = 0; j < CONV1_FILTERS; ++j) {
            model.conv2_weights[i][j].resize(CONV2_KERNEL);
            for (int k = 0; k < CONV2_KERNEL; ++k) {
                model.conv2_weights[i][j][k] = dist(rng);
            }
        }
    }
    model.conv2_bias.resize(CONV2_FILTERS);
    for (int i = 0; i < CONV2_FILTERS; ++i) {
        model.conv2_bias[i] = dist(rng);
    }
    model.conv2_output_size = model.conv1_output_size - CONV2_KERNEL + 1;
    
    // Dense1 weights
    int dense1_input_size = CONV2_FILTERS * model.conv2_output_size;
    model.dense1_weights.resize(DENSE1_SIZE);
    for (int i = 0; i < DENSE1_SIZE; ++i) {
        model.dense1_weights[i].resize(dense1_input_size);
        for (int j = 0; j < dense1_input_size; ++j) {
            model.dense1_weights[i][j] = dist(rng);
        }
    }
    model.dense1_bias.resize(DENSE1_SIZE);
    for (int i = 0; i < DENSE1_SIZE; ++i) {
        model.dense1_bias[i] = dist(rng);
    }
    
    // Dense2 weights
    model.dense2_weights.resize(OUTPUT_SIZE);
    for (int i = 0; i < OUTPUT_SIZE; ++i) {
        model.dense2_weights[i].resize(DENSE1_SIZE);
        for (int j = 0; j < DENSE1_SIZE; ++j) {
            model.dense2_weights[i][j] = dist(rng);
        }
    }
    model.dense2_bias.resize(OUTPUT_SIZE);
    for (int i = 0; i < OUTPUT_SIZE; ++i) {
        model.dense2_bias[i] = dist(rng);
    }
    
    return model;
}

// ReLU activation
inline float relu(float x) {
    return std::max(0.0f, x);
}

// 1D Convolution
void conv1d(const std::vector<float>& input, int input_size, int input_channels,
            const std::vector<std::vector<std::vector<float>>>& weights,
            const std::vector<float>& bias,
            int num_filters, int kernel_size,
            std::vector<float>& output) {
    int output_size = input_size - kernel_size + 1;
    output.resize(num_filters * output_size);
    
    for (int f = 0; f < num_filters; ++f) {
        for (int i = 0; i < output_size; ++i) {
            float sum = bias[f];
            for (int c = 0; c < input_channels; ++c) {
                for (int k = 0; k < kernel_size; ++k) {
                    sum += input[c * input_size + i + k] * weights[f][c][k];
                }
            }
            output[f * output_size + i] = relu(sum);
        }
    }
}

// 1D Convolution for first layer (single channel)
void conv1d_first(const std::vector<float>& input,
                  const std::vector<std::vector<float>>& weights,
                  const std::vector<float>& bias,
                  int num_filters, int kernel_size,
                  std::vector<float>& output) {
    int output_size = input.size() - kernel_size + 1;
    output.resize(num_filters * output_size);
    
    for (int f = 0; f < num_filters; ++f) {
        for (int i = 0; i < output_size; ++i) {
            float sum = bias[f];
            for (int k = 0; k < kernel_size; ++k) {
                sum += input[i + k] * weights[f][k];
            }
            output[f * output_size + i] = relu(sum);
        }
    }
}

// Dense layer
void dense(const std::vector<float>& input,
           const std::vector<std::vector<float>>& weights,
           const std::vector<float>& bias,
           std::vector<float>& output,
           bool apply_relu = true) {
    int output_size = weights.size();
    output.resize(output_size);
    
    for (int i = 0; i < output_size; ++i) {
        float sum = bias[i];
        for (size_t j = 0; j < input.size(); ++j) {
            sum += input[j] * weights[i][j];
        }
        output[i] = apply_relu ? relu(sum) : sum;
    }
}

// Softmax
void softmax(std::vector<float>& x) {
    float max_val = *std::max_element(x.begin(), x.end());
    float sum = 0.0f;
    for (float& val : x) {
        val = std::exp(val - max_val);
        sum += val;
    }
    for (float& val : x) {
        val /= sum;
    }
}

// CNN Forward pass (batch=1)
void cnn_inference(const CNNModel& model, const std::vector<float>& input,
                   std::vector<float>& output) {
    // Conv1
    std::vector<float> conv1_out;
    conv1d_first(input, model.conv1_weights, model.conv1_bias,
                 CONV1_FILTERS, CONV1_KERNEL, conv1_out);
    
    // Conv2
    std::vector<float> conv2_out;
    conv1d(conv1_out, model.conv1_output_size, CONV1_FILTERS,
           model.conv2_weights, model.conv2_bias,
           CONV2_FILTERS, CONV2_KERNEL, conv2_out);
    
    // Dense1
    std::vector<float> dense1_out;
    dense(conv2_out, model.dense1_weights, model.dense1_bias, dense1_out, true);
    
    // Dense2
    dense(dense1_out, model.dense2_weights, model.dense2_bias, output, false);
    
    // Softmax
    softmax(output);
}

void benchmark_cnn_inference() {
    std::cout << "\n====================================" << std::endl;
    std::cout << "CNN Inference CPU Benchmark" << std::endl;
    std::cout << "====================================" << std::endl;
    
    // Model info
    std::cout << "\n----- Network Architecture -----" << std::endl;
    std::cout << "Input: [" << INPUT_SIZE << "] 1D signal" << std::endl;
    std::cout << "Conv1D(" << CONV1_FILTERS << " filters, kernel=" << CONV1_KERNEL << ") -> ReLU" << std::endl;
    std::cout << "Conv1D(" << CONV2_FILTERS << " filters, kernel=" << CONV2_KERNEL << ") -> ReLU" << std::endl;
    std::cout << "Dense(" << DENSE1_SIZE << ") -> ReLU" << std::endl;
    std::cout << "Dense(" << OUTPUT_SIZE << ") -> Softmax" << std::endl;
    
    // Initialize model
    std::cout << "\nInitializing model..." << std::endl;
    CNNModel model = initialize_model();
    
    // Calculate model size
    int total_params = CONV1_FILTERS * CONV1_KERNEL + CONV1_FILTERS;
    total_params += CONV2_FILTERS * CONV1_FILTERS * CONV2_KERNEL + CONV2_FILTERS;
    total_params += DENSE1_SIZE * (CONV2_FILTERS * model.conv2_output_size) + DENSE1_SIZE;
    total_params += OUTPUT_SIZE * DENSE1_SIZE + OUTPUT_SIZE;
    
    std::cout << "Total parameters: " << total_params << std::endl;
    std::cout << "Model size: " << (total_params * sizeof(float)) / 1024.0 << " KB" << std::endl;
    
    // Generate random input
    std::mt19937 rng(123);
    std::uniform_real_distribution<float> input_dist(-1.0f, 1.0f);
    std::vector<float> input(INPUT_SIZE);
    for (int i = 0; i < INPUT_SIZE; ++i) {
        input[i] = input_dist(rng);
    }
    
    std::vector<float> output;
    
    // Warmup
    for (int i = 0; i < 10; ++i) {
        cnn_inference(model, input, output);
    }
    
    // Benchmark - batch=1 streaming
    std::cout << "\n----- Batch=1 Streaming Inference -----" << std::endl;
    const int num_inferences = 1000;
    
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < num_inferences; ++i) {
        cnn_inference(model, input, output);
    }
    auto end = std::chrono::high_resolution_clock::now();
    
    std::chrono::duration<double> elapsed = end - start;
    double avg_latency_ms = (elapsed.count() * 1000.0) / num_inferences;
    double inferences_per_sec = num_inferences / elapsed.count();
    
    // Calculate FLOPs
    long long flops = 0;
    // Conv1: output_size * filters * kernel_size * 2 (MAC)
    flops += (long long)model.conv1_output_size * CONV1_FILTERS * CONV1_KERNEL * 2;
    // Conv2
    flops += (long long)model.conv2_output_size * CONV2_FILTERS * CONV1_FILTERS * CONV2_KERNEL * 2;
    // Dense1
    flops += (long long)DENSE1_SIZE * (CONV2_FILTERS * model.conv2_output_size) * 2;
    // Dense2
    flops += (long long)OUTPUT_SIZE * DENSE1_SIZE * 2;
    
    double gflops_per_sec = (flops * inferences_per_sec) / 1e9;
    
    // Print results
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "Inferences:        " << num_inferences << std::endl;
    std::cout << "Total time:        " << elapsed.count() * 1000.0 << " ms" << std::endl;
    std::cout << "Avg latency:       " << avg_latency_ms << " ms/inference" << std::endl;
    std::cout << "Throughput:        " << inferences_per_sec << " inferences/sec" << std::endl;
    std::cout << "Compute:           " << gflops_per_sec << " GFLOP/s" << std::endl;
    std::cout << "FLOPs/inference:   " << flops << std::endl;
    
    // Output verification
    std::cout << "\n----- Output Sample -----" << std::endl;
    std::cout << "Class probabilities: [";
    for (int i = 0; i < OUTPUT_SIZE; ++i) {
        std::cout << output[i];
        if (i < OUTPUT_SIZE - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;
    
    int predicted_class = std::max_element(output.begin(), output.end()) - output.begin();
    std::cout << "Predicted class: " << predicted_class << std::endl;
}

int main() {
    std::cout << "====================================" << std::endl;
    std::cout << "Workload E: CNN/ML Inference Block" << std::endl;
    std::cout << "Architecture: CPU" << std::endl;
    std::cout << "====================================" << std::endl;
    
    benchmark_cnn_inference();
    
    return 0;
}
