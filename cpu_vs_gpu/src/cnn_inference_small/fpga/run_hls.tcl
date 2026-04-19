# Vitis HLS TCL script for CNN inference kernel synthesis

# Create project
open_project -reset cnn_hls_project

# Add design files
add_files cnn_kernel.cpp
add_files cnn_kernel.h

# Add testbench (if available)
# add_files -tb host.cpp

# Set top function
set_top cnn_inference_kernel

# Create solution
open_solution -reset "solution1"

# Set FPGA part (Alveo U250 example)
set_part {xcu250-figd2104-2L-e}

# Create clock with 300 MHz target (3.33 ns period)
create_clock -period 3.33 -name default

# Optimization directives
# These are already specified in the source via pragmas:
# - DATAFLOW for task-level pipelining
# - PIPELINE for loop pipelining
# - BIND_STORAGE for memory mapping

# Additional solution-level config
config_compile -pipeline_loops 64

# Run C simulation (functional verification)
# csim_design

# Run synthesis
csynth_design

# Run C/RTL co-simulation (optional, very slow)
# cosim_design -rtl verilog -trace_level all

# Export RTL (optional)
# export_design -format ip_catalog

# Generate reports
puts "========================================"
puts "HLS Synthesis Complete"
puts "========================================"
puts "Check reports in: cnn_hls_project/solution1/syn/report/"
puts ""
puts "Key metrics to review:"
puts "  - Latency (cycles)"
puts "  - Interval (II)"
puts "  - Resource utilization (LUT, FF, BRAM, DSP)"
puts "  - Clock period achieved"
puts ""

# Close project
close_project

exit
