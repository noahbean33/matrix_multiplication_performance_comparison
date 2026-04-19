cmake_minimum_required(VERSION 3.10)

project(StreamingStencil2D_CPU CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(stencil_cpu main.cpp)

# Add compiler-specific optimization flags for Release builds
if(MSVC)
    # /O2 is the standard optimization flag for release builds in MSVC
    # /fp:fast allows for more aggressive floating-point optimizations
    target_compile_options(stencil_cpu PRIVATE $<$<CONFIG:Release>:/O2 /fp:fast>)
else()
    # -O3 and -march=native are aggressive optimization flags for GCC/Clang
    target_compile_options(stencil_cpu PRIVATE $<$<CONFIG:Release>:-O3 -march=native -ffast-math>)
endif()

install(TARGETS stencil_cpu DESTINATION bin)
