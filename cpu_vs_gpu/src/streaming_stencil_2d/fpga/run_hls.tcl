# Vitis HLS TCL script for Streaming Stencil kernel synthesis

# Create project
open_project -reset stencil_hls_project

# Add design files
add_files stencil_kernel.cpp
add_files stencil_kernel.h

# Set top function (streaming version - the key optimization!)
set_top stencil_kernel_streaming

# Create solution
open_solution -reset "solution1"

# Set FPGA part (Alveo U250 example)
set_part {xcu250-figd2104-2L-e}

# Create clock with 300 MHz target (3.33 ns period)
create_clock -period 3.33 -name default

# Optimization directives already in source via pragmas:
# - PIPELINE II=1 for inner loop (1 point per cycle!)
# - ARRAY_PARTITION for line buffers (parallel access)
# - BIND_STORAGE to place buffers in BRAM

# Run synthesis
csynth_design

puts "========================================"
puts "HLS Synthesis Complete - Streaming Stencil"
puts "========================================"
puts "Check reports in: stencil_hls_project/solution1/syn/report/"
puts ""
puts "Expected results:"
puts "  - Pipeline II: 1 cycle (inner loop)"
puts "  - Throughput: ~1 point per cycle"
puts "  - Latency: ~width cycles (line buffer initialization)"
puts "  - BRAM: ~2 rows for line buffers"
puts ""
puts "This demonstrates FPGA's streaming advantage:"
puts "  ✓ Sequential access (burst I/O)"
puts "  ✓ Line buffers enable stencil without random access"
puts "  ✓ Fully pipelined architecture"
puts "  ✓ Minimal DDR traffic"
puts ""
puts "This is FPGA's HOME TURF!"
puts "========================================"

# Close project
close_project

exit
