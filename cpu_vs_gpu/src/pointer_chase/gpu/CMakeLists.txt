cmake_minimum_required(VERSION 3.18)

project(PointerChase_GPU CUDA CXX)

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
add_executable(pointer_chase_gpu main.cu)

# Set CUDA architectures
set_target_properties(pointer_chase_gpu PROPERTIES
    CUDA_ARCHITECTURES "61;75;86"
)

# Optimization flags
target_compile_options(pointer_chase_gpu PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        --use_fast_math
        -O3
    >
)

install(TARGETS pointer_chase_gpu DESTINATION bin)
