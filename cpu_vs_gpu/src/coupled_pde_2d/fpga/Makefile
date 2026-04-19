# Makefile for Coupled PDE FPGA Implementation
# Target: Xilinx Vitis HLS / Vivado

# Configuration
KERNEL_NAME = pde_kernel_advect_streaming
PLATFORM ?= xilinx_u250_gen3x16_xdma_4_1_202210_1
TARGET ?= sw_emu
FREQ ?= 300

# Directories
BUILD_DIR = ./build
KERNEL_SRC = pde_kernel.cpp
HOST_SRC = host.cpp
KERNEL_HDR = pde_kernel.h

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
	@echo "This workload demonstrates FPGA's BRAM PRESSURE:"
	@echo "  - 5 fields × N² × 4 bytes"
	@echo "  - 256²: ~1.25 MB (feasible)"
	@echo "  - 512²: ~5 MB (BRAM pressure)"
	@echo "  - 1024²: ~20 MB (exceeds BRAM!)"

# Software emulation (no FPGA tools required)
sw_emu: $(BUILD_DIR)/host_sw

$(BUILD_DIR)/host_sw: $(HOST_SRC) $(KERNEL_SRC) $(KERNEL_HDR)
	@mkdir -p $(BUILD_DIR)
	g++ $(HOST_CFLAGS) -o $@ $(HOST_SRC) $(KERNEL_SRC) -I.

# Hardware emulation
hw_emu: TARGET = hw_emu
hw_emu: $(BUILD_DIR)/pde.xclbin $(BUILD_DIR)/host

# Hardware build
hw: TARGET = hw
hw: $(BUILD_DIR)/pde.xclbin $(BUILD_DIR)/host

# Compile kernel to xo
$(BUILD_DIR)/$(KERNEL_NAME).xo: $(KERNEL_SRC) $(KERNEL_HDR)
	@mkdir -p $(BUILD_DIR)
	$(VPP) $(VPP_OPTS) $(KERNEL_OPTS) -c -k $(KERNEL_NAME) -o $@ $(KERNEL_SRC)

# Link kernel to xclbin
$(BUILD_DIR)/pde.xclbin: $(BUILD_DIR)/$(KERNEL_NAME).xo
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
	XCL_EMULATION_MODE=hw_emu $(BUILD_DIR)/host --hw $(BUILD_DIR)/pde.xclbin

# Run on hardware
run_hw: hw
	$(BUILD_DIR)/host --hw $(BUILD_DIR)/pde.xclbin

# HLS synthesis report
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

# BRAM analysis
bram_analysis:
	@echo "========================================"
	@echo "FPGA BRAM Capacity Analysis"
	@echo "========================================"
	@echo ""
	@echo "Alveo U250 BRAM Resources:"
	@echo "  Total BRAM: ~96 MB (2688 × 36 KB)"
	@echo "  Usable per kernel: ~20-40 MB"
	@echo ""
	@echo "Coupled PDE Memory Requirements:"
	@echo "  5 fields (u, v, p, T, div) × FP32"
	@echo ""
	@echo "Grid | Per Field | Total  | Fits?"
	@echo "-----|-----------|--------|-------"
	@echo "256² |  0.25 MB  | 1.25 MB| ✓ YES"
	@echo "512² |  1.00 MB  | 5.00 MB| ⚠ TIGHT"
	@echo "1024²|  4.00 MB  | 20.0 MB| ✗ NO"
	@echo ""
	@echo "Additional Challenge: Jacobi iterations"
	@echo "  - Need ping-pong buffers (2× memory)"
	@echo "  - 100 iterations with inter-dependencies"
	@echo "  - Cannot stream (random access)"
	@echo ""
	@echo "Conclusion:"
	@echo "  This workload shows FPGA's BRAM bottleneck!"
	@echo "========================================"

# Performance estimate
estimate:
	@echo "========================================"
	@echo "FPGA Coupled PDE - Expected Performance"
	@echo "========================================"
	@echo "Target Platform: $(PLATFORM)"
	@echo "Target Frequency: $(FREQ) MHz"
	@echo ""
	@echo "Expected Performance by Grid Size:"
	@echo ""
	@echo "256² (BRAM-based):"
	@echo "  - Advection: ~250 M pts/s (streaming)"
	@echo "  - Diffusion: ~250 M pts/s (streaming)"
	@echo "  - Jacobi: ~100 M pts/s (iterative)"
	@echo "  - Overall: ~120 M pts/s"
	@echo ""
	@echo "512² (Hybrid BRAM/DDR):"
	@echo "  - Advection: ~200 M pts/s"
	@echo "  - Diffusion: ~200 M pts/s"
	@echo "  - Jacobi: ~60 M pts/s (DDR pressure)"
	@echo "  - Overall: ~80 M pts/s"
	@echo ""
	@echo "1024² (DDR-only):"
	@echo "  - Advection: ~150 M pts/s"
	@echo "  - Diffusion: ~150 M pts/s"
	@echo "  - Jacobi: ~40 M pts/s (DDR-bound)"
	@echo "  - Overall: ~50 M pts/s"
	@echo ""
	@echo "GPU Comparison (for reference):"
	@echo "  - Overall: ~200-300 M pts/s (all sizes)"
	@echo "  - GPU WINS for this workload!"
	@echo ""
	@echo "Why FPGA struggles:"
	@echo "  ✗ BRAM capacity insufficient"
	@echo "  ✗ Jacobi iterations don't pipeline well"
	@echo "  ✗ Multi-field pressure"
	@echo "  ✗ Cannot achieve streaming dataflow"
	@echo "========================================"
