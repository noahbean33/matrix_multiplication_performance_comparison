cmake_minimum_required(VERSION 3.10)

project(SpMV_CPU CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(spmv_cpu main.cpp)

# Add compiler-specific optimization flags for Release builds
if(MSVC)
    target_compile_options(spmv_cpu PRIVATE $<$<CONFIG:Release>:/O2 /fp:fast>)
else()
    target_compile_options(spmv_cpu PRIVATE $<$<CONFIG:Release>:-O3 -march=native -ffast-math>)
endif()

install(TARGETS spmv_cpu DESTINATION bin)
