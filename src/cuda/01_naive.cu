/*naive implementation of MM*/
/*created for cuda couese on udemy*/
#include <iostream>
#include <cuda_runtime.h>
#include <cstdlib>

__global__ void mm_naive(float*A, float*B , float*C, int N){

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N) {
        int sum = 0;
        for (int k = 0; k < N; k++) {
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    } 
}


void cpu_mm_naive(const float* A, const float* B, float* C, int N)
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
    const int N=512;

    float* h_A = new float[N*N];
    float* h_B = new float[N*N];
    float* h_C = new float[N*N];
    float* h_C_gpu = new float[N*N];

    float *d_A,*d_B,*d_C;
    cudaMalloc((void**)&d_A,N*N*sizeof(float));
    cudaMalloc((void**)&d_B,N*N*sizeof(float));
    cudaMalloc((void**)&d_C,N*N*sizeof(float));

    for(int i=0;i<N*N;i++){
        h_A[i]=std::rand() % 10;
        h_B[i]=std::rand() % 10;
        h_C[i]=0;
        h_C_gpu[i]=0;
    }


    cpu_mm_naive(h_A,h_B,h_C,N);

    cudaMemcpy(d_A,h_A,N*N*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(d_B,h_B,N*N*sizeof(float),cudaMemcpyHostToDevice);

    int block_size=16;
    dim3 threadsperblock(block_size,block_size);
    dim3 numBlocks((N+block_size-1)/block_size,(N+block_size-1)/block_size);

    mm_naive<<<numBlocks,threadsperblock>>>(d_A,d_B,d_C,N);

    cudaMemcpy(h_C_gpu,d_C,N*N*sizeof(float),cudaMemcpyDeviceToHost);

    long temp=0;
    for(int i=0;i<N*N;i++){
        temp += std::abs(h_C_gpu[i]-h_C[i]);
    }
    if(temp==0) 
     std::cout<<"results are correct\n";
    else
    std::cout<<"wrong results\n";

    



    delete[] h_A;
    delete[] h_B;
    delete[] h_C;
    delete[] h_C_gpu;

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

}