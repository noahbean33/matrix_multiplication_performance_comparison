/******************************************************************************
 * File: mm_float4_only.cu
 *
 * Demonstrates:
 *   - Tiled matrix multiplication using shared memory
 *   - Vectorized float4 loads (assuming alignment is correct)
 *   - No pinned memory usage (uses normal new[]/delete[] + cudaMalloc/cudaMemcpy)
 *
 * Compile:
 *   nvcc mm_float4_only.cu -o mm_float4_only -std=c++11
 ******************************************************************************/

#include <iostream>
#include <cuda_runtime.h>
#include <cstdlib>
#include <cmath>

// For simplicity, we pick N=512 (a multiple of 16)
static const int N = 512;
// We'll keep tile size "16×16" conceptually, but blockDim.x=4 => each thread loads 4 columns.
static const int TILE_WIDTH = 16;  // tile is 16 wide, 16 tall

//------------------------------------------------------------------------------
// CPU naive mm
//------------------------------------------------------------------------------
void cpu_mm_naive(const float* A, const float* B, float* C, int n)
{
    for (int i = 0; i < n; i++){
        for (int j = 0; j < n; j++){
            float sum = 0.0f;
            for (int k = 0; k < n; k++){
                sum += A[i*n + k] * B[k*n + j];
            }
            C[i*n + j] = sum;
        }
    }
}

//------------------------------------------------------------------------------
// Kernel: Tiled + float4 loads
//   blockDim: (4,16) => 4 threads in x, each loading 4 columns => 16 columns total
//                         16 threads in y => 16 rows
//------------------------------------------------------------------------------
__global__ void mmTiledVec4(const float* __restrict__ A,
                            const float* __restrict__ B,
                            float*       __restrict__ C,
                            int n)
{
    // Shared memory for one tile (16×16) of A and one tile (16×16) of B
    __shared__ float sA[TILE_WIDTH][TILE_WIDTH];
    __shared__ float sB[TILE_WIDTH][TILE_WIDTH];

    // Identify which row & column(s) this thread will handle
    // blockIdx.x, blockIdx.y => which tile
    // threadIdx.x in [0..3], threadIdx.y in [0..15]
    int Row = blockIdx.y * TILE_WIDTH + threadIdx.y;
    // Each thread covers 4 columns:
    //   baseCol = blockIdx.x*16 + threadIdx.x*4
    int baseCol = blockIdx.x * TILE_WIDTH + (threadIdx.x * 4);

    // We'll accumulate partial sums for these 4 columns in a small array
    float sum[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    // how many 16×16 tiles we must iterate over in the K dimension
    int numTiles = n / TILE_WIDTH;  // assume n is multiple of 16

    // For each tile in [0..numTiles-1]
    for (int t = 0; t < numTiles; t++)
    {
        // --------------------------------------------------
        // 1. Load sub-tile of A and B into shared memory
        //    Each thread loads a float4 from A, and a float4 from B
        // --------------------------------------------------

        // A
        //sA[threadIdx.y][threadIdx.x] = A[row * n + (t * TILE_SIZE + threadIdx.x)];
        {   //
            // global col for A => t*TILE_WIDTH + (threadIdx.x*4 .. +3)
            int globalACol = t*TILE_WIDTH + (threadIdx.x * 4);
            const float* srcA = A + (Row*n + globalACol);
            float* dstA = &sA[threadIdx.y][threadIdx.x*4];

            float4 vecA = *reinterpret_cast<const float4*>(srcA);
            *reinterpret_cast<float4*>(dstA) = vecA;
        }

        //sB[threadIdx.y][threadIdx.x] = B[(t * TILE_SIZE + threadIdx.y) * N + col];
        // B
        {
            // global row for B => t*TILE_WIDTH + threadIdx.y
            int globalBRow = t*TILE_WIDTH + threadIdx.y;
            // columns => baseCol..baseCol+3
            const float* srcB = B + (globalBRow*n + baseCol);
            float* dstB = &sB[threadIdx.y][threadIdx.x*4];

            float4 vecB = *reinterpret_cast<const float4*>(srcB);
            *reinterpret_cast<float4*>(dstB) = vecB;
        }

        __syncthreads();

        // --------------------------------------------------
        // 2. Compute partial sums
        // --------------------------------------------------
        for(int k = 0; k < TILE_WIDTH; k++){
            float aVal = sA[threadIdx.y][k];
            float4 bVal4 = *reinterpret_cast<float4*>(&sB[k][threadIdx.x*4]);
            sum[0] += aVal * bVal4.x;
            sum[1] += aVal * bVal4.y;
            sum[2] += aVal * bVal4.z;
            sum[3] += aVal * bVal4.w;
        }

        __syncthreads();
    }

    // --------------------------------------------------
    // 3. Write out final partial sums
    // --------------------------------------------------
    if(Row < n){
        // We skip boundary checks for columns. If baseCol+3 < n => safe
        float* outC = C + (Row*n + baseCol);
        outC[0] = sum[0];
        outC[1] = sum[1];
        outC[2] = sum[2];
        outC[3] = sum[3];
    }
}

//------------------------------------------------------------------------------
// Main
//------------------------------------------------------------------------------
int main()
{
    // 1. Allocate host memory (regular/paged)
    float* h_A = new float[N*N];
    float* h_B = new float[N*N];
    float* h_C = new float[N*N];
    float* h_Cgpu = new float[N*N];

    // 2. Initialize
    srand(0);
    for(int i=0; i<N*N; i++){
        h_A[i] = float(rand()%10);
        h_B[i] = float(rand()%10);
        h_C[i] = 0.0f;
        h_Cgpu[i] = 0.0f;
    }

    // CPU reference
    cpu_mm_naive(h_A, h_B, h_C, N);

    // 3. Allocate device memory
    float *d_A, *d_B, *d_C;
    size_t nBytes = N*N*sizeof(float);
    cudaMalloc((void**)&d_A, nBytes);
    cudaMalloc((void**)&d_B, nBytes);
    cudaMalloc((void**)&d_C, nBytes);

    // 4. Copy data from Host to Device (synchronous)
    cudaMemcpy(d_A, h_A, nBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, nBytes, cudaMemcpyHostToDevice);

    // 5. Launch the vectorized kernel
    // blockDim => (4,16). => 4 in x => 4 columns per thread, 16 in y => 16 rows
    dim3 block(4,16);
    dim3 grid(N/TILE_WIDTH, N/TILE_WIDTH);
    mmTiledVec4<<<grid, block>>>(d_A, d_B, d_C, N);
    cudaDeviceSynchronize();

    // 6. Copy result back
    cudaMemcpy(h_Cgpu, d_C, nBytes, cudaMemcpyDeviceToHost);

    // 7. Check correctness
    long long diff = 0;
    for(int i=0; i<N*N; i++){
        diff += (long long)fabs(h_Cgpu[i] - h_C[i]);
    }
    if(diff==0) {
        std::cout << "Results match!\n";
    } else {
        std::cout << "Mismatch, diff=" << diff << "\n";
    }

    // Cleanup
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;
    delete[] h_Cgpu;
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}
