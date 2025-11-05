#include <stdio.h>    // For printf
#include <stdlib.h>   // For malloc, free, rand, srand
#include <time.h>     // For time, clock

// Allocates a random n x n matrix of doubles between 0 and 1,
double** generate_random_matrix(int n) {
    // Allocate memory for an array of row pointers (double*), one for each row
    double** M = (double**)malloc(n * sizeof(double*));

    // For each row i
    for (int i = 0; i < n; i++) {
        // Allocate memory for n doubles in this row
        M[i] = (double*)malloc(n * sizeof(double));
        
        // Fill each element in the row with a random double in [0, 1]
        for (int j = 0; j < n; j++) {
            M[i][j] = (double)rand() / (double)RAND_MAX;
        }
    }
    return M; // Return the pointer to the matrix
}

// Performs matrix multiplication C = A * B for two n x n matrices 
void matrix_multiply(double** A, double** B, double** C, int n) {
    // Loop over each row of A
    for (int i = 0; i < n; i++) {
        // Loop over each column of B
        for (int j = 0; j < n; j++) {
            double sum = 0.0; // Initialize the accumulator for the dot product
            
            // Compute the dot product of row i of A and column j of B
            for (int k = 0; k < n; k++) {
                sum += A[i][k] * B[k][j];
            }
            
            // Store the computed value in C[i][j]
            C[i][j] = sum;
        }
    }
}

// Frees the dynamically allocated memory for an n x n matrix M.
void free_matrix(double** M, int n) {
    // Free each row of the matrix
    for (int i = 0; i < n; i++) {
        free(M[i]);
    }
    // Free the array of row pointers
    free(M);
}

// Tests matrix multiplication for various matrix sizes and measures performance.
int main() {
    // Define the range of matrix sizes and the increment step for testing
    int min_size = 2;    // Smallest matrix size to test
    int max_size = 1002; // Largest matrix size to test
    int step = 50;       // Increment step for matrix size

    // Seed the random number generator with the current time
    srand((unsigned)time(NULL));

    // Print the CSV header to standard output for easy parsing
    printf("MatrixSize,TimeSeconds\n");

    // Iterate over matrix sizes from min_size to max_size
    for (int n = min_size; n <= max_size; n += step) {
        // Generate two random n x n matrices A and B
        double** A = generate_random_matrix(n);
        double** B = generate_random_matrix(n);

        // Allocate memory for the result matrix C
        double** C = (double**)malloc(n * sizeof(double*));
        for (int i = 0; i < n; i++) {
            C[i] = (double*)calloc(n, sizeof(double));
        }

        // Measure the time taken for matrix multiplication
        clock_t start = clock();          // Start timing
        matrix_multiply(A, B, C, n);      // Perform the multiplication
        clock_t end = clock();            // End timing

        // Calculate elapsed time in seconds
        double elapsed = (double)(end - start) / CLOCKS_PER_SEC;

        // Print the matrix size and elapsed time in CSV format
        printf("%d,%.6f\n", n, elapsed);

        // Free the dynamically allocated matrices to avoid memory leaks
        free_matrix(A, n);
        free_matrix(B, n);
        free_matrix(C, n);
    }

    return 0;
}
