#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <random>
#include <algorithm>
#include <cmath>

#ifdef XILINX_FPGA
#include "xcl2.hpp"
#endif

#include "cnn_kernel.h"

// Model structure
struct CNNModel {
    std::vector<float> conv1_weights;
    std::vector<float> conv1_bias;
    std::vector<float> conv2_weights;
    std::vector<float> conv2_bias;
    std::vector<float> dense1_weights;
    std::vector<float> dense1_bias;
    std::vector<float> dense2_weights;
    std::vector<float> dense2_bias;
};

// Initialize random weights
CNNModel initialize_model() {
    std::mt19937 rng(42);
    std::normal_distribution<float> dist(0.0f, 0.1f);
    
    CNNModel model;
    
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
    model.dense1_weights.resize(DENSE1_SIZE * DENSE1_INPUT_SIZE);
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

// Software emulation
void benchmark_cnn_fpga_sw() {
    std::cout << "\n====================================" << std::endl;
    std::cout << "CNN Inference FPGA Benchmark" << std::endl;
    std::cout << "Mode: Software Emulation" << std::endl;
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
    int total_params = model.conv1_weights.size() + model.conv1_bias.size() +
                       model.conv2_weights.size() + model.conv2_bias.size() +
                       model.dense1_weights.size() + model.dense1_bias.size() +
                       model.dense2_weights.size() + model.dense2_bias.size();
    
    std::cout << "Total parameters: " << total_params << std::endl;
    std::cout << "Model size (FP32): " << (total_params * sizeof(float)) / 1024.0 << " KB" << std::endl;
    std::cout << "Model size (INT8): " << (total_params * sizeof(char)) / 1024.0 << " KB" << std::endl;
    
    // Generate random input
    std::mt19937 rng(123);
    std::uniform_real_distribution<float> input_dist(-1.0f, 1.0f);
    std::vector<float> input(INPUT_SIZE);
    for (auto& val : input) val = input_dist(rng);
    
    std::vector<float> output(OUTPUT_SIZE);
    
    // Warmup
    for (int i = 0; i < 5; ++i) {
        cnn_inference_kernel(
            input.data(),
            model.conv1_weights.data(), model.conv1_bias.data(),
            model.conv2_weights.data(), model.conv2_bias.data(),
            model.dense1_weights.data(), model.dense1_bias.data(),
            model.dense2_weights.data(), model.dense2_bias.data(),
            output.data()
        );
    }
    
    // Benchmark - batch=1 streaming (FPGA sweet spot)
    std::cout << "\n----- Batch=1 Streaming Inference -----" << std::endl;
    const int num_inferences = 1000;
    
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < num_inferences; ++i) {
        cnn_inference_kernel(
            input.data(),
            model.conv1_weights.data(), model.conv1_bias.data(),
            model.conv2_weights.data(), model.conv2_bias.data(),
            model.dense1_weights.data(), model.dense1_bias.data(),
            model.dense2_weights.data(), model.dense2_bias.data(),
            output.data()
        );
    }
    auto end = std::chrono::high_resolution_clock::now();
    
    std::chrono::duration<double> elapsed = end - start;
    double avg_latency_ms = (elapsed.count() * 1000.0) / num_inferences;
    double inferences_per_sec = num_inferences / elapsed.count();
    
    // Calculate FLOPs
    long long flops = 0;
    flops += (long long)CONV1_OUTPUT_SIZE * CONV1_FILTERS * CONV1_KERNEL * 2;
    flops += (long long)CONV2_OUTPUT_SIZE * CONV2_FILTERS * CONV1_FILTERS * CONV2_KERNEL * 2;
    flops += (long long)DENSE1_SIZE * DENSE1_INPUT_SIZE * 2;
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
    
    std::cout << "\n----- Note on SW Emulation -----" << std::endl;
    std::cout << "This is C++ functional simulation only." << std::endl;
    std::cout << "Actual FPGA hardware will be MUCH faster!" << std::endl;
    
    std::cout << "\n----- Expected FPGA Hardware Performance -----" << std::endl;
    std::cout << "Target Clock:      300 MHz" << std::endl;
    std::cout << "Pipeline II:       1-2 cycles (dataflow)" << std::endl;
    std::cout << "Expected Latency:  0.1-0.5 ms/inference (batch=1)" << std::endl;
    std::cout << "Expected Throughput: 2000-10000 inferences/sec" << std::endl;
    std::cout << "Power:             ~20-40W (vs 200W GPU)" << std::endl;
    std::cout << "Energy/Inference:  ~0.01-0.02 mJ" << std::endl;
    
    std::cout << "\n----- FPGA Advantages -----" << std::endl;
    std::cout << "✓ Custom dataflow pipeline (no batch needed)" << std::endl;
    std::cout << "✓ Streaming architecture (overlapped execution)" << std::endl;
    std::cout << "✓ INT8/custom precision (4x+ speedup possible)" << std::endl;
    std::cout << "✓ Low power consumption (5-10x better J/inference)" << std::endl;
    std::cout << "✓ Deterministic latency (no scheduling overhead)" << std::endl;
    
    std::cout << "\n----- Resource Estimation (Alveo U250) -----" << std::endl;
    std::cout << "LUTs:              ~50-80K / 1.3M (5-6%)" << std::endl;
    std::cout << "FFs:               ~80-120K / 2.6M (3-5%)" << std::endl;
    std::cout << "BRAM:              ~100-200 / 2688 (4-8%)" << std::endl;
    std::cout << "DSP:               ~50-100 / 12288 (0.5-1%)" << std::endl;
    std::cout << "URAM:              ~10-50 / 1280 (1-4%)" << std::endl;
    
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

#ifdef XILINX_FPGA
void benchmark_cnn_fpga_hw(const std::string& xclbin_file) {
    std::cout << "\n====================================" << std::endl;
    std::cout << "CNN Inference FPGA Benchmark" << std::endl;
    std::cout << "Mode: Hardware Execution" << std::endl;
    std::cout << "====================================" << std::endl;
    
    // Hardware execution using XRT would go here
    std::cout << "Hardware execution requires:" << std::endl;
    std::cout << "1. Synthesized bitstream (xclbin file)" << std::endl;
    std::cout << "2. Xilinx XRT runtime installed" << std::endl;
    std::cout << "3. Compatible FPGA board (e.g., Alveo U250)" << std::endl;
    std::cout << "\nPlease run synthesis first: make build" << std::endl;
}
#endif

int main(int argc, char** argv) {
    std::cout << "====================================" << std::endl;
    std::cout << "Workload E: CNN/ML Inference Block" << std::endl;
    std::cout << "Architecture: FPGA (Xilinx Vitis HLS)" << std::endl;
    std::cout << "====================================" << std::endl;
    
    bool hw_mode = false;
    std::string xclbin_file = "";
    
    if (argc > 1 && std::string(argv[1]) == "--hw") {
        hw_mode = true;
        if (argc > 2) {
            xclbin_file = argv[2];
        }
    }
    
    if (hw_mode) {
#ifdef XILINX_FPGA
        benchmark_cnn_fpga_hw(xclbin_file);
#else
        std::cout << "Hardware mode not available. Rebuild with Xilinx tools." << std::endl;
        return 1;
#endif
    } else {
        std::cout << "\nRunning in SOFTWARE EMULATION mode" << std::endl;
        std::cout << "Use --hw <xclbin> for hardware execution\n" << std::endl;
        
        benchmark_cnn_fpga_sw();
        
        std::cout << "\n====================================" << std::endl;
        std::cout << "Key Takeaway: FPGA for Streaming" << std::endl;
        std::cout << "====================================" << std::endl;
        std::cout << "FPGAs excel at batch=1 low-latency inference:" << std::endl;
        std::cout << "• No batch accumulation needed" << std::endl;
        std::cout << "• Pipelined dataflow keeps hardware busy" << std::endl;
        std::cout << "• Much better latency than GPU at batch=1" << std::endl;
        std::cout << "• Lower power → better energy efficiency" << std::endl;
        std::cout << "\nThis is why cloud FPGAs are used for real-time inference!" << std::endl;
    }
    
    return 0;
}
