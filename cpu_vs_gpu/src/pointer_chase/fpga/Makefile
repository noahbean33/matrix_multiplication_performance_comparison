# Makefile for Pointer Chase FPGA Implementation
# Target: Xilinx Vitis HLS / Vivado

# Configuration
KERNEL_NAME = pointer_chase_kernel_baseline
PLATFORM ?= xilinx_u250_gen3x16_xdma_4_1_202210_1
TARGET ?= sw_emu
FREQ ?= 300

# Directories
BUILD_DIR = ./build
KERNEL_SRC = pointer_chase_kernel.cpp
HOST_SRC = host.cpp
KERNEL_HDR = pointer_chase_kernel.h

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
	@echo "This workload demonstrates:"
	@echo "  - Predictable pattern: FPGA custom prefetch helps"
	@echo "  - Random pattern: Everyone is DDR-bound"
	@echo "  - Shows when FPGAs STOP looking magical"

# Software emulation (no FPGA tools required)
sw_emu: $(BUILD_DIR)/host_sw

$(BUILD_DIR)/host_sw: $(HOST_SRC) $(KERNEL_SRC) $(KERNEL_HDR)
	@mkdir -p $(BUILD_DIR)
	g++ $(HOST_CFLAGS) -o $@ $(HOST_SRC) $(KERNEL_SRC) -I.

# Hardware emulation
hw_emu: TARGET = hw_emu
hw_emu: $(BUILD_DIR)/pointer_chase.xclbin $(BUILD_DIR)/host

# Hardware build
hw: TARGET = hw
hw: $(BUILD_DIR)/pointer_chase.xclbin $(BUILD_DIR)/host

# Compile kernel to xo
$(BUILD_DIR)/$(KERNEL_NAME).xo: $(KERNEL_SRC) $(KERNEL_HDR)
	@mkdir -p $(BUILD_DIR)
	$(VPP) $(VPP_OPTS) $(KERNEL_OPTS) -c -k $(KERNEL_NAME) -o $@ $(KERNEL_SRC)

# Link kernel to xclbin
$(BUILD_DIR)/pointer_chase.xclbin: $(BUILD_DIR)/$(KERNEL_NAME).xo
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
	XCL_EMULATION_MODE=hw_emu $(BUILD_DIR)/host --hw $(BUILD_DIR)/pointer_chase.xclbin

# Run on hardware
run_hw: hw
	$(BUILD_DIR)/host --hw $(BUILD_DIR)/pointer_chase.xclbin

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

# Performance estimate
estimate:
	@echo "========================================"
	@echo "FPGA Pointer Chase - Key Insights"
	@echo "========================================"
	@echo ""
	@echo "Predictable Pattern:"
	@echo "  Latency: ~30-80 ns/hop (with prefetch)"
	@echo "  Resources: 5-10K LUTs, 10-50 BRAM"
	@echo "  FPGA can pipeline prefetch operations"
	@echo ""
	@echo "Random Pattern:"
	@echo "  Latency: ~80-150 ns/hop (DDR latency)"
	@echo "  Serial dependency limits performance"
	@echo "  No amount of custom logic can fix this!"
	@echo ""
	@echo "This shows when FPGAs STOP looking magical."
	@echo "========================================"
