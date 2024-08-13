#include <iostream>
#include <vector>
#include <chrono>
#include <cuda_runtime.h>

__global__ void vectorAddKernel(const float *A, const float *B, float *C, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N)
    {
        C[i] = A[i] + B[i];
    }
}

void vectorAdd(const std::vector<float> &A, const std::vector<float> &B, std::vector<float> &C, int N)
{
    float *d_A, *d_B, *d_C;
    size_t size = N * sizeof(float);

    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    cudaMemcpy(d_A, A.data(), size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B.data(), size, cudaMemcpyHostToDevice);

    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;
    vectorAddKernel<<<numBlocks, blockSize>>>(d_A, d_B, d_C, N);

    cudaMemcpy(C.data(), d_C, size, cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

int main()
{
    int N = 1 << 28; // 1M elements
    std::vector<float> A(N, 1.0f);
    std::vector<float> B(N, 2.0f);
    std::vector<float> C(N);

    auto start = std::chrono::high_resolution_clock::now();
    vectorAdd(A, B, C, N);
    cudaDeviceSynchronize(); // Ensure all work on the GPU is finished
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<float, std::milli> duration = end - start;
    std::cout << "Time taken: " << duration.count() << " ms" << std::endl;

    return 0;
}
