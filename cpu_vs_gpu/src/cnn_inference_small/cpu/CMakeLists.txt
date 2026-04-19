cmake_minimum_required(VERSION 3.10)

project(CNN_Inference_CPU CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(cnn_cpu main.cpp)

# Add compiler-specific optimization flags for Release builds
if(MSVC)
    target_compile_options(cnn_cpu PRIVATE $<$<CONFIG:Release>:/O2 /fp:fast /arch:AVX2>)
else()
    target_compile_options(cnn_cpu PRIVATE $<$<CONFIG:Release>:-O3 -march=native -ffast-math>)
endif()

install(TARGETS cnn_cpu DESTINATION bin)
