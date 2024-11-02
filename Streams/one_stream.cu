#include <iostream>
#include <cuda_runtime.h>

// Kernel to perform basic element-wise addition and multiplication
__global__ void performOperations(float *d_a, float *d_b, float *d_result, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N)
    {
        d_result[idx] = (d_a[idx] + d_b[idx]) * d_a[idx];
    }
}

int main()
{
    int N = 1 << 20; // Number of elements
    size_t size = N * sizeof(float);
    const int numIterations = 50;

    // Allocate memory on the host (CPU)
    float *h_a = (float *)malloc(size);
    float *h_b = (float *)malloc(size);
    float *h_result = (float *)malloc(size);

    // Initialize host arrays with some values
    for (int i = 0; i < N; ++i)
    {
        h_a[i] = static_cast<float>(i);
        h_b[i] = static_cast<float>(i) * 0.5f;
    }

    // Allocate memory on the device (GPU)
    float *d_a, *d_b, *d_result;
    cudaMalloc((void **)&d_a, size);
    cudaMalloc((void **)&d_b, size);
    cudaMalloc((void **)&d_result, size);

    // Repeat HtoD and kernel execution for 50 iterations
    for (int i = 0; i < numIterations; ++i)
    {
        // Transfer data from host to device (H2D)
        cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
        cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);

        // Launch kernel to perform operations (using 256 threads per block)
        int threadsPerBlock = 256;
        int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
        performOperations<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_result, N);
    }

    // After 50 iterations, transfer the final result from device to host (D2H)
    cudaMemcpy(h_result, d_result, size, cudaMemcpyDeviceToHost);

    // Verify result (print a few values)
    for (int i = 0; i < 10; ++i)
    {
        std::cout << "Result[" << i << "] = " << h_result[i] << std::endl;
    }

    // Free device and host memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_result);
    free(h_a);
    free(h_b);
    free(h_result);

    return 0;
}
