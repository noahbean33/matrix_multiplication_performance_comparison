`timescale 1ns/1ns

module matrix_multiplier_tb;

    // Task to initialize a matrix with random values
    task automatic init_matrix(input int n, output logic [31:0] matrix []);
        int i, j;
        matrix = new[n * n]; // Allocate memory for the dynamic array
        for (i = 0; i < n; i++) begin
            for (j = 0; j < n; j++) begin
                matrix[i * n + j] = $urandom() % 100; // Random value between 0 and 99
            end
        end
    endtask

    // Task to multiply two matrices with simulated computation delay
    task automatic multiply_matrices(input int n,
                                      input logic [31:0] matrix_a [],
                                      input logic [31:0] matrix_b [],
                                      output logic [31:0] result []);
        int i, j, k;
        result = new[n * n]; // Allocate memory for the result matrix
        for (i = 0; i < n; i++) begin
            for (j = 0; j < n; j++) begin
                result[i * n + j] = 0;
                for (k = 0; k < n; k++) begin
                    result[i * n + j] += matrix_a[i * n + k] * matrix_b[k * n + j];
                    #1; // Simulate computation delay for each multiplication
                end
            end
        end
    endtask

    // Testbench logic
    initial begin
        int n;
        real start_time, end_time;
        logic [31:0] matrix_a [], matrix_b [], result [];

      for (n = 700; n <= 1002; n += 2) begin // Increment n by 100
            // Initialize matrices
            init_matrix(n, matrix_a);
            init_matrix(n, matrix_b);

            // Measure multiplication time
            start_time = $realtime; // Capture simulation time
            multiply_matrices(n, matrix_a, matrix_b, result);
            end_time = $realtime; // Capture simulation time after multiplication

            // Print results
            $display("Matrix size: %0dx%0d, Time taken: %0.2f ns", n, n, end_time - start_time);

            // Deallocate matrices
            matrix_a.delete();
            matrix_b.delete();
            result.delete();
        end

        $finish;
    end
endmodule
