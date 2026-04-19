# Vitis HLS TCL script for SpMV kernel synthesis

# Create project
open_project -reset spmv_hls_project

# Add design files
add_files spmv_kernel.cpp
add_files spmv_kernel.h

# Add testbench (if available)
# add_files -tb host.cpp

# Set top function
set_top spmv_csr_kernel

# Create solution
open_solution -reset "solution1"

# Set FPGA part (Alveo U250 example)
set_part {xcu250-figd2104-2L-e}

# Create clock with 300 MHz target (3.33 ns period)
create_clock -period 3.33 -name default

# Synthesis directives (can be added here or in source)
# These are already in the source code via pragmas

# Run C simulation
# csim_design

# Run synthesis
csynth_design

# Run C/RTL co-simulation (optional, slow)
# cosim_design

# Export RTL
# export_design -format ip_catalog

# Close project
close_project

puts "HLS synthesis complete. Check spmv_hls_project/solution1/syn/report/ for reports."
exit
