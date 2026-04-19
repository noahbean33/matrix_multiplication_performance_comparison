# Vitis HLS TCL script for Pointer Chase kernel synthesis

# Create project
open_project -reset pointer_chase_hls_project

# Add design files
add_files pointer_chase_kernel.cpp
add_files pointer_chase_kernel.h

# Set top function (baseline version)
set_top pointer_chase_kernel_baseline

# Create solution
open_solution -reset "solution1"

# Set FPGA part (Alveo U250 example)
set_part {xcu250-figd2104-2L-e}

# Create clock with 300 MHz target (3.33 ns period)
create_clock -period 3.33 -name default

# Key observation: The main loop has a loop-carried dependency
# so it CANNOT be pipelined with II=1
# HLS will report: "Cannot schedule operation due to dependence"

# Run synthesis
csynth_design

puts "========================================"
puts "HLS Synthesis Complete"
puts "========================================"
puts "Check reports in: pointer_chase_hls_project/solution1/syn/report/"
puts ""
puts "Expected results:"
puts "  - Loop II: Cannot pipeline (loop-carried dependency)"
puts "  - Latency: ~num_hops * (DDR_latency + compute)"
puts "  - Throughput: Low (serial execution)"
puts ""
puts "This is the FUNDAMENTAL limitation:"
puts "Each hop depends on the previous result."
puts "No hardware architecture can parallelize this!"
puts ""

# Close project
close_project

exit
