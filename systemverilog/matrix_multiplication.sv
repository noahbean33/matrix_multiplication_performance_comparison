//////////////////////////////////////////////////////////////////////////////////
// Company:        Oregon State University
// Engineer:       Noah Bean
// 
// Create Date:    12/13/2024
// Design Name:    Matrix Multiplication 
// Module Name:    matrix_multiplication
// Project Name:   Matrix_Multiplication_SystemVerilog_ECE472_Project
// Target Devices: Nexys A7-100T
// Tool Versions:  Vivado 2024.2
// Description: 
//   This is a module matrix multiplication. 
//
// Dependencies: 
//   - NA
//
// Revision:
//   Revision 0.01 - File Created
//   
// 
// Additional Comments:
//   Ensure that the synthesis and simulation libraries are properly configured
//   in the Vivado project for successful simulation.
//
// License: 
//   MIT
//////////////////////////////////////////////////////////////////////////////////
`timescale 1ns/1ns

module matrix_multiplication (
    input logic Clock,             // Clock signal, drives the sequential logic
    input logic reset,             // Active-high reset signal
    input logic Enable,            // Enable signal; matrix multiplication starts when high
    input logic [71:0] A,          // Input matrix A, represented as a 72-bit 1D array
    input logic [71:0] B,          // Input matrix B, represented as a 72-bit 1D array
    output logic [71:0] C,         // Output matrix C, represented as a 72-bit 1D array
    output logic done              // Indicates when multiplication is complete
);

    // Internal 2D arrays to store individual matrix elements
    logic signed [7:0] matA [2:0][2:0]; // 3x3 matrix for A
    logic signed [7:0] matB [2:0][2:0]; // 3x3 matrix for B
    logic signed [7:0] matC [2:0][2:0]; // 3x3 matrix for C

    // Control variables for loops and operations
    integer i, j, k;           // Indices for row (i), column (j), and inner product (k)
    logic first_cycle;         // Tracks the first clock cycle after Enable goes high
    logic end_of_mult;         // Becomes high when multiplication is complete
    logic signed [15:0] temp;  // Temporary variable for intermediate product calculations

    // Always block triggered on positive edge of Clock or when reset is asserted
    always_ff @(posedge Clock or posedge reset) begin
        if (reset) begin
            // Reset all variables to initial state
            i = 0; j = 0; k = 0;               // Reset loop indices
            temp = 0;                          // Clear temporary product variable
            first_cycle = 1;                   // Indicate first cycle after reset
            end_of_mult = 0;                   // Clear multiplication completion flag
            done = 0;                          // Clear done signal
            // Initialize matrices to zero
            foreach (matA[i, j]) begin
                matA[i][j] = 8'sd0;
                matB[i][j] = 8'sd0;
                matC[i][j] = 8'sd0;
            end
        end else if (Enable) begin
            if (first_cycle) begin
                // Convert 1D array inputs A and B into 2D matrices matA and matB
                foreach (matA[i, j]) begin
                    matA[i][j] = A[(i * 3 + j) * 8 +: 8]; // Extract 8 bits for each element
                    matB[i][j] = B[(i * 3 + j) * 8 +: 8];
                    matC[i][j] = 8'sd0; // Initialize matC elements to zero
                end
                // Reset control flags and indices for multiplication
                first_cycle = 0;
                end_of_mult = 0;
                temp = 0;
                i = 0; j = 0; k = 0;
            end else if (!end_of_mult) begin
                // Perform matrix multiplication
                // Calculate the product of matA[i][k] and matB[k][j], then accumulate
                temp = matA[i][k] * matB[k][j];  // Multiply elements
                matC[i][j] += temp[7:0];        // Add lower 8 bits of product to result
                // Manage nested loops for matrix traversal
                if (k == 2) begin               // Inner loop completes when k reaches 2
                    k = 0;                      // Reset k
                    if (j == 2) begin           // Middle loop completes when j reaches 2
                        j = 0;                  // Reset j
                        if (i == 2) begin       // Outer loop completes when i reaches 2
                            i = 0;              // Reset i
                            end_of_mult = 1;    // Indicate multiplication is complete
                        end else begin
                            i++;                // Move to the next row
                        end
                    end else begin
                        j++;                    // Move to the next column
                    end
                end else begin
                    k++;                        // Increment inner loop index
                end
            end else begin
                // Convert 2D result matrix matC back to a 1D array for output
                foreach (matC[i, j]) begin
                    C[(i * 3 + j) * 8 +: 8] = matC[i][j];
                end
                done = 1;  // Signal that the multiplication is complete
            end
        end
    end
endmodule
