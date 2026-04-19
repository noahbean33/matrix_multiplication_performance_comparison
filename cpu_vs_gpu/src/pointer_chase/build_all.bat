@echo off
REM Convenience script to build all Pointer Chase implementations on Windows

echo ========================================
echo Building Pointer Chase Benchmark Suite
echo ========================================

REM CPU Build
echo.
echo [1/3] Building CPU version...
cd cpu
if exist build rmdir /s /q build
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . --config Release
cd ..\..
echo [SUCCESS] CPU build complete
echo.

REM GPU Build (check for CUDA)
echo [2/3] Building GPU version...
where nvcc >nul 2>nul
if %ERRORLEVEL% EQU 0 (
    cd gpu
    if exist build rmdir /s /q build
    mkdir build
    cd build
    cmake .. -DCMAKE_BUILD_TYPE=Release
    cmake --build . --config Release
    cd ..\..
    echo [SUCCESS] GPU build complete
) else (
    echo [SKIPPED] CUDA not found, skipping GPU build
    echo           Install CUDA Toolkit to build GPU version
)
echo.

REM FPGA Build (software emulation - requires MinGW/g++)
echo [3/3] Building FPGA software emulation...
where g++ >nul 2>nul
if %ERRORLEVEL% EQU 0 (
    cd fpga
    if exist build rmdir /s /q build
    mkdir build
    g++ -O3 -std=c++17 -o build/host_sw.exe host.cpp pointer_chase_kernel.cpp -I.
    cd ..
    echo [SUCCESS] FPGA software emulation build complete
) else (
    echo [SKIPPED] g++ not found, skipping FPGA emulation build
    echo           Install MinGW or MSYS2 to build FPGA emulation
)
echo.

echo ========================================
echo Build Summary
echo ========================================
echo CPU:  cpu\build\Release\pointer_chase_cpu.exe
where nvcc >nul 2>nul
if %ERRORLEVEL% EQU 0 (
    echo GPU:  gpu\build\Release\pointer_chase_gpu.exe
) else (
    echo GPU:  [not built - CUDA not available]
)
where g++ >nul 2>nul
if %ERRORLEVEL% EQU 0 (
    echo FPGA: fpga\build\host_sw.exe ^(software emulation^)
) else (
    echo FPGA: [not built - g++ not available]
)
echo.
echo Run individual benchmarks:
echo   cd cpu\build\Release ^&^& pointer_chase_cpu.exe
echo   cd gpu\build\Release ^&^& pointer_chase_gpu.exe
echo   cd fpga\build ^&^& host_sw.exe
echo.
echo WARNING: Key Insight:
echo   This shows when FPGAs STOP looking magical!
echo   Predictable: FPGA custom prefetch helps
echo   Random: Everyone is DDR-bound (~100ns latency)
echo.
pause
