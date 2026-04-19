cmake_minimum_required(VERSION 3.18)

project(StreamingStencil2D_GPU CUDA CXX)

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
add_executable(stencil_gpu main.cu)

# Set CUDA architectures
set_target_properties(stencil_gpu PROPERTIES
    CUDA_ARCHITECTURES "61;75;86"
)

# Optimization flags
target_compile_options(stencil_gpu PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        --use_fast_math
        -O3
    >
)

install(TARGETS stencil_gpu DESTINATION bin)
