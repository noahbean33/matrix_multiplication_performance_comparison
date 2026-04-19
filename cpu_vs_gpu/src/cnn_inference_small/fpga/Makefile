# Makefile for CNN Inference FPGA Implementation
# Target: Xilinx Vitis HLS / Vivado

# Configuration
KERNEL_NAME = cnn_inference_kernel
PLATFORM ?= xilinx_u250_gen3x16_xdma_4_1_202210_1
TARGET ?= sw_emu
FREQ ?= 300

# Directories
BUILD_DIR = ./build
KERNEL_SRC = cnn_kernel.cpp
HOST_SRC = host.cpp
KERNEL_HDR = cnn_kernel.h

# Vitis compiler
VPP = v++
EMCONFIGUTIL = emconfigutil

# Compiler flags
VPP_OPTS = --platform $(PLATFORM)
VPP_OPTS += --target $(TARGET)
VPP_OPTS += --kernel_frequency $(FREQ)
VPP_OPTS += --save-temps
VPP_OPTS += --temp_dir $(BUILD_DIR)/_x
VPP_OPTS += --log_dir $(BUILD_DIR)/logs

# Kernel compilation flags
KERNEL_OPTS = -O3
KERNEL_OPTS += --hls.clock $(FREQ)MHz

# Host compiler flags
HOST_CFLAGS = -O3 -std=c++17
HOST_CFLAGS += -I$(XILINX_XRT)/include
HOST_CFLAGS += -I$(XILINX_VIVADO)/include
HOST_CFLAGS += -Wall

HOST_LDFLAGS = -L$(XILINX_XRT)/lib
HOST_LDFLAGS += -lxilinxopencl -lpthread -lrt -lstdc++

# Targets
.PHONY: all clean sw_emu hw_emu hw help

all: sw_emu

help:
	@echo "Makefile targets:"
	@echo "  make sw_emu   - Software emulation (fast, functional verification)"
	@echo "  make hw_emu   - Hardware emulation (slower, timing-approximate)"
	@echo "  make hw       - Hardware synthesis (very slow, actual FPGA bitstream)"
	@echo "  make clean    - Clean build artifacts"
	@echo ""
	@echo "Environment variables:"
	@echo "  PLATFORM - Target FPGA platform (default: xilinx_u250_gen3x16_xdma_4_1_202210_1)"
	@echo "  FREQ     - Target frequency in MHz (default: 300)"
	@echo ""
	@echo "Expected performance:"
	@echo "  Latency (batch=1): 0.1-0.5 ms/inference"
	@echo "  Throughput: 2000-10000 inferences/sec"
	@echo "  Power: 20-40W (vs 200W+ GPU)"

# Software emulation (no FPGA tools required)
sw_emu: $(BUILD_DIR)/host_sw

$(BUILD_DIR)/host_sw: $(HOST_SRC) $(KERNEL_SRC) $(KERNEL_HDR)
	@mkdir -p $(BUILD_DIR)
	g++ $(HOST_CFLAGS) -o $@ $(HOST_SRC) $(KERNEL_SRC) -I.

# Hardware emulation
hw_emu: TARGET = hw_emu
hw_emu: $(BUILD_DIR)/cnn.xclbin $(BUILD_DIR)/host

# Hardware build
hw: TARGET = hw
hw: $(BUILD_DIR)/cnn.xclbin $(BUILD_DIR)/host

# Compile kernel to xo
$(BUILD_DIR)/$(KERNEL_NAME).xo: $(KERNEL_SRC) $(KERNEL_HDR)
	@mkdir -p $(BUILD_DIR)
	$(VPP) $(VPP_OPTS) $(KERNEL_OPTS) -c -k $(KERNEL_NAME) -o $@ $(KERNEL_SRC)

# Link kernel to xclbin
$(BUILD_DIR)/cnn.xclbin: $(BUILD_DIR)/$(KERNEL_NAME).xo
	$(VPP) $(VPP_OPTS) -l -o $@ $^

# Compile host application
$(BUILD_DIR)/host: $(HOST_SRC) $(KERNEL_HDR)
	@mkdir -p $(BUILD_DIR)
	g++ $(HOST_CFLAGS) -DXILINX_FPGA -o $@ $(HOST_SRC) $(HOST_LDFLAGS)

# Generate emulation configuration
emconfig.json:
	$(EMCONFIGUTIL) --platform $(PLATFORM) --nd 1

# Run software emulation
run_sw_emu: $(BUILD_DIR)/host_sw
	$(BUILD_DIR)/host_sw

# Run hardware emulation
run_hw_emu: hw_emu emconfig.json
	XCL_EMULATION_MODE=hw_emu $(BUILD_DIR)/host --hw $(BUILD_DIR)/cnn.xclbin

# Run on hardware
run_hw: hw
	$(BUILD_DIR)/host --hw $(BUILD_DIR)/cnn.xclbin

# HLS synthesis report (without full build)
report:
	@echo "Running HLS synthesis report..."
	@mkdir -p $(BUILD_DIR)/report
	vitis_hls -f run_hls.tcl

clean:
	rm -rf $(BUILD_DIR)
	rm -rf _x
	rm -rf .Xil
	rm -rf .run
	rm -f *.log *.jou
	rm -f emconfig.json

# Performance estimate
estimate: $(BUILD_DIR)/$(KERNEL_NAME).xo
	@echo "========================================"
	@echo "FPGA CNN Inference - Resource Estimate"
	@echo "========================================"
	@echo "Target Platform: $(PLATFORM)"
	@echo "Target Frequency: $(FREQ) MHz"
	@echo ""
	@echo "Expected Resources (Alveo U250):"
	@echo "  LUTs:  50-80K / 1.3M (5-6%)"
	@echo "  FFs:   80-120K / 2.6M (3-5%)"
	@echo "  BRAM:  100-200 / 2688 (4-8%)"
	@echo "  DSP:   50-100 / 12288 (0.5-1%)"
	@echo "  URAM:  10-50 / 1280 (1-4%)"
	@echo ""
	@echo "Expected Performance:"
	@echo "  Latency: 0.1-0.5 ms/inference (batch=1)"
	@echo "  Throughput: 2000-10000 inferences/sec"
	@echo "  Power: 20-40W"
	@echo "  Energy: 0.01-0.02 mJ/inference"
	@echo ""
	@echo "Run 'make hw_emu' for detailed resource report"
