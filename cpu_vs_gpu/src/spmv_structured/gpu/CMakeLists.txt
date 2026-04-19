cmake_minimum_required(VERSION 3.18)

project(SpMV_GPU CUDA CXX)

# Enable CUDA
enable_language(CUDA)

# Set C++ and CUDA standards
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Find CUDA
find_package(CUDAToolkit REQUIRED)

# Create executable
add_executable(spmv_gpu main.cu)

# Set CUDA architectures (adjust based on your GPU)
# Common architectures: 
# - 61: GTX 1080, GTX 1070, etc.
# - 75: RTX 2080, RTX 2070, etc.
# - 86: RTX 3090, RTX 3080, etc.
# - 89: RTX 4090, RTX 4080, etc.
set_target_properties(spmv_gpu PROPERTIES
    CUDA_ARCHITECTURES "61;75;86"
)

# Optimization flags
target_compile_options(spmv_gpu PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        --use_fast_math
        -O3
    >
)

install(TARGETS spmv_gpu DESTINATION bin)
