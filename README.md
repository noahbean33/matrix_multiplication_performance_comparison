# # **Matrix Multiplication Performance Comparison**

This repository contains the implementation and analysis of matrix multiplication using Python, C, and SystemVerilog. It explores the performance, scalability, and trade-offs between these methods for various matrix sizes.

---

## **Overview**
Matrix multiplication is a fundamental algorithm widely used in scientific computing, signal pprocessing, and machine learning. This project evaluates different implementations of matrix multiplication by comparing their execution times in order to assess the quantitive and qualitative trade-offs involved. 

---

## **Features**
1. **Implementations:**
   - **Python:** Basic triple-loop and optimized NumPy implementations.
   - **C:** Unoptimized and compiler-optimized (`-O3`) implementations.
   - **SystemVerilog:** Hardware-based implementation.

2. **Performance Analysis:**
   - Comparison across varying matrix sizes (2×2 to 1002×1002).
   - Execution time measurements and scalability evaluation in units of seconds.

3. **Trade-offs:**
   - Ease of implementation versus computational efficiency.
   - Hardware complexity versus software complexity.

4. **Challenges Addressed:**
   - Developing scalable HDL implementations for arbitrary matrix sizes.
   - Overcoming hardware synthesis limitations.

---

## **Files and Directory Structure**

### **Source Files**
- `python/`: Contains Python scripts for basic and NumPy-based implementations.
- `c/`: Includes C Program.
- `systemverilog/`: Contains the SystemVerilog modules and testbenches for HDL-based implementation.

### **Reports**
- `Matrix_Multiplication_Performance_Comparison_ECE_472_Project.pdf`: Detailed analysis of implementation results, challenges, and lessons learned.
- `results/`: Contains raw data and plots comparing execution times.

---

## **How to Run**

### **1. Prerequisites**
- Python 3 with NumPy installed.
- GCC or any standard C compiler.
- Xilinx Vivado, EDA Playgrounnd, or any Verilog simulation tool.

### **2. Steps**
1. Clone the repository:
```bash
   git clone https://github.com/noahbean33/Matrix_Multiplication_Performance_Comparison.git
   cd Matrix_Multiplication_Performance_Comparison
```

   ### **2. Steps**   
#### **Run Python implementations**:
   ```bash
cd python
python3 matrix_multiplication_numpy.py
python3 matrix_multiplication_python.py
```

#### Run C implementations:
   ```bash
cd c
gcc -o matrix_multiplication matrix_multiplication.c
./basic
gcc -O3 -o matrix_multiplication matrix_multiplication.c
./optimized
   ```

#### Run SystemVerilog implementations:
   ```bash
cd SystemVerilog
xrun -sv -timescale 1ns/1ns -access +rw matrix_multiplier_tb.sv (Cadence Xcelium 23.09)
   ```

# Additional Details

**Assuming Cadence Xcelium 23.09 on EDA Playground**
**matrix_multiplication.sv and matrix_multiplication_tb.sv can be set up in a project on Xilinx Vivado**

## Results

The performance evaluation highlighted the following:

- **Python (Basic):** Easy to implement but slow for large matrices.
- **NumPy:** Leveraged optimized C libraries for the fastest performance.
- **C (Optimized):** Balanced speed and control, suitable for most use cases.
- **SystemVerilog:** Demonstrating excellent scalability and speed but with higher complexity.

For detailed graphs and analysis, see `Matrix_Multiplication_Performance_Comparison_ECE_472_Project.pdf` in the repository.

---

## Lessons Learned

- Hardware-level parallelism is critical for scaling computationally intensive tasks.
- Simulation tools like Xilinx Vivado streamline rapid prototyping.
- Compiler optimizations significantly improve software performance with minimal effort.

---

## Future Work

- Implement floating-point support in HDL.
- Deploy on advanced FPGA hardware to confirm simulation results.
- Explore hybrid software-hardware approaches for computational efficiency.

---

## Contributions

Contributions to enhance the implementations or add new features are welcome!

---

## License

This project is licensed under the MIT License. See the `LICENSE` file for more details.

---

## Acknowledgments

Thanks to the authors of the referenced books, tools, and tutorials that supported this project.
