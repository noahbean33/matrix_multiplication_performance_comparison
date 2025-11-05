#include <iostream>
#include <cuda_runtime.h>

#define N 1024            // Assume N is a multiple of TILE_SIZE
#define TILE_SIZE 16     // Could also use 32 for some GPUs

// GPU Kernel: Tiled MM (assuming N % TILE_SIZE == 0)
__global__ void matrixMulTiled(const float* A, const float* B, float* C)
{
    __shared__ float sA[TILE_SIZE][TILE_SIZE];
    __shared__ float sB[TILE_SIZE][TILE_SIZE];

    // Calculate this thread's row and column
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    // Each thread will accumulate the result in 'sum'
    float sum = 0.0f;

    // Number of tiles we need to iterate over the k dimension
    // Since N is multiple of TILE_SIZE, this is N / TILE_SIZE
    int numTiles = N / TILE_SIZE;

    for (int t = 0; t < numTiles; t++) {
        // Load one tile of A into shared memory
        sA[threadIdx.y][threadIdx.x] = A[row * N + (t * TILE_SIZE + threadIdx.x)];
        // Load one tile of B into shared memory
        sB[threadIdx.y][threadIdx.x] = B[(t * TILE_SIZE + threadIdx.y) * N + col];

        // Synchronize to make sure the tile is loaded
        __syncthreads();

        // Compute partial sums for this tile
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += sA[threadIdx.y][k] * sB[k][threadIdx.x];
        }

        // Synchronize before loading the next tile
        __syncthreads();
    }

    // Write the result to global memory
    C[row * N + col] = sum;
}
void cpu_mm_naive( float* A,  float* B, float* C)
{
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0;
            for (int k = 0; k < N; k++) {
                sum += A[i * N + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}
int main()
{

    // 1. Allocate Host Memory (simple example)
    float *h_A = new float[N*N];
    float *h_B = new float[N*N];
    float *h_C = new float[N*N];
    float *h_C_gpu = new float[N*N];

    // 2. Initialize Input Matrices
    for (int i = 0; i < N*N; i++) {
        h_A[i] = static_cast<float>(i % 10); // arbitrary
        h_B[i] = static_cast<float>((i * 2) % 10);
    }

    // 3. Allocate Device Memory
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, N*N*sizeof(float));
    cudaMalloc(&d_B, N*N*sizeof(float));
    cudaMalloc(&d_C, N*N*sizeof(float));

    // 4. Copy Data from Host to Device
    cudaMemcpy(d_A, h_A, N*N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, N*N*sizeof(float), cudaMemcpyHostToDevice);

    // 5. Launch Tiled Kernel
    dim3 threads(TILE_SIZE, TILE_SIZE);
    dim3 blocks(N / TILE_SIZE, N / TILE_SIZE);
    cpu_mm_naive(h_A,h_B,h_C);

    matrixMulTiled<<<blocks, threads>>>(d_A, d_B, d_C);

    // 6. Copy Result Back
    cudaMemcpy(h_C_gpu, d_C, N*N*sizeof(float), cudaMemcpyDeviceToHost);

    // 7. (Optional) Verify or Print Some Results
    long temp=0;
    for(int i=0;i<N*N;i++){
        temp += std::abs(h_C_gpu[i]-h_C[i]);
    }
    if(temp==0) 
     std::cout<<"results are correct\n";
    else
    std::cout<<"wrong results\n";
    // 8. Cleanup
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}
