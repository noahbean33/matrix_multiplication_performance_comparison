# Vitis HLS TCL script for Coupled PDE kernel synthesis

# Create project
open_project -reset pde_hls_project

# Add design files
add_files pde_kernel.cpp
add_files pde_kernel.h

# Set top function (streaming advection - the only part that works well)
set_top pde_kernel_advect_streaming

# Create solution
open_solution -reset "solution1"

# Set FPGA part (Alveo U250 example)
set_part {xcu250-figd2104-2L-e}

# Create clock with 300 MHz target (3.33 ns period)
create_clock -period 3.33 -name default

# Key observations for this workload:
# - Advection/Diffusion: Can pipeline with II=1 (streaming)
# - Jacobi iteration: CANNOT pipeline across iterations
# - BRAM requirement: 5 fields × N² × 4 bytes
#   - 256²: 1.25 MB (feasible)
#   - 512²: 5.0 MB (tight)
#   - 1024²: 20 MB (EXCEEDS BRAM!)

# Run synthesis
csynth_design

puts "========================================"
puts "HLS Synthesis Complete - Coupled PDE"
puts "========================================"
puts "Check reports in: pde_hls_project/solution1/syn/report/"
puts ""
puts "Expected results for streaming advection:"
puts "  - Pipeline II: 1-2 cycles"
puts "  - Throughput: ~200-300 M points/sec"
puts "  - BRAM usage: Line buffers only (~few KB)"
puts ""
puts "BRAM Analysis for Full Solver:"
puts "  256²:  1.25 MB  → Feasible (need ~35 BRAM blocks)"
puts "  512²:  5.0 MB   → Tight (need ~140 BRAM blocks)"
puts "  1024²: 20.0 MB  → EXCEEDS typical allocation!"
puts ""
puts "Jacobi Iteration Challenge:"
puts "  - 100 iterations with dependency"
puts "  - Cannot pipeline ACROSS iterations"
puts "  - Within iteration: II=1 achievable"
puts "  - Overall: 100× slower than streaming ops"
puts ""
puts "This demonstrates FPGA's BRAM PRESSURE bottleneck:"
puts "  ✗ Multi-field storage exceeds on-chip capacity"
puts "  ✗ Iterative solvers don't pipeline well"
puts "  ✗ Cannot stream (random access pattern)"
puts "  ✗ DDR bandwidth becomes bottleneck"
puts ""
puts "GPU is likely FASTER for this workload!"
puts "========================================"

# Optionally synthesize the BRAM version (256² only)
puts ""
puts "Synthesizing BRAM version (small grids only)..."
puts ""

# Create second solution for BRAM version
open_solution -reset "solution_bram"
set_top pde_kernel_bram
set_part {xcu250-figd2104-2L-e}
create_clock -period 3.33 -name default

csynth_design

puts ""
puts "BRAM Version Results:"
puts "  Expected BRAM usage: ~50-80 blocks (for 256²)"
puts "  Works ONLY for grids ≤ 256²"
puts "  Larger grids: Resource exhaustion!"
puts ""

# Close project
close_project

exit
