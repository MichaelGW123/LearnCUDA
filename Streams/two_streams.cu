#include <iostream>
#include <cuda_runtime.h>

// CUDA error checking macro
#define CUDA_CHECK(call)                                                               \
    {                                                                                  \
        cudaError_t err = call;                                                        \
        if (err != cudaSuccess)                                                        \
        {                                                                              \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << std::endl; \
            std::cerr << "Error: " << cudaGetErrorString(err) << std::endl;            \
            exit(EXIT_FAILURE);                                                        \
        }                                                                              \
    }

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
    float *d_a[2], *d_b[2], *d_result[2];
    CUDA_CHECK(cudaMalloc((void **)&d_a[0], size));
    CUDA_CHECK(cudaMalloc((void **)&d_b[0], size));
    CUDA_CHECK(cudaMalloc((void **)&d_result[0], size));
    CUDA_CHECK(cudaMalloc((void **)&d_a[1], size));
    CUDA_CHECK(cudaMalloc((void **)&d_b[1], size));
    CUDA_CHECK(cudaMalloc((void **)&d_result[1], size));

    // Create CUDA streams
    cudaStream_t stream1, stream2;
    CUDA_CHECK(cudaStreamCreate(&stream1));
    CUDA_CHECK(cudaStreamCreate(&stream2));

    // Repeat HtoD and kernel execution for 50 iterations, alternating streams
    for (int i = 0; i < numIterations; ++i)
    {
        int streamIdx = i % 2; // Alternates between 0 and 1

        // Use stream1 or stream2 based on streamIdx
        cudaStream_t currentStream = (streamIdx == 0) ? stream1 : stream2;

        // Transfer data from host to device (H2D) using the current stream
        CUDA_CHECK(cudaMemcpyAsync(d_a[streamIdx], h_a, size, cudaMemcpyHostToDevice, currentStream));
        CUDA_CHECK(cudaMemcpyAsync(d_b[streamIdx], h_b, size, cudaMemcpyHostToDevice, currentStream));

        // Launch kernel to perform operations using the current stream
        int threadsPerBlock = 256;
        int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
        performOperations<<<blocksPerGrid, threadsPerBlock, 0, currentStream>>>(d_a[streamIdx], d_b[streamIdx], d_result[streamIdx], N);
        std::cout << "Launched kernel iteration " << i << std::endl;

        // Check for kernel launch errors
        CUDA_CHECK(cudaGetLastError());
    }

    // Synchronize both streams before final D2H transfer
    CUDA_CHECK(cudaStreamSynchronize(stream1));
    CUDA_CHECK(cudaStreamSynchronize(stream2));

    // Transfer the final result from the device to the host (D2H) from the last used stream
    CUDA_CHECK(cudaMemcpy(h_result, d_result[(numIterations - 1) % 2], size, cudaMemcpyDeviceToHost));

    // Verify result (print a few values)
    for (int i = 0; i < 10; ++i)
    {
        std::cout << "Result[" << i << "] = " << h_result[i] << std::endl;
    }

    // Free device and host memory
    CUDA_CHECK(cudaFree(d_a[0]));
    CUDA_CHECK(cudaFree(d_b[0]));
    CUDA_CHECK(cudaFree(d_result[0]));
    CUDA_CHECK(cudaFree(d_a[1]));
    CUDA_CHECK(cudaFree(d_b[1]));
    CUDA_CHECK(cudaFree(d_result[1]));
    free(h_a);
    free(h_b);
    free(h_result);

    // Destroy CUDA streams
    CUDA_CHECK(cudaStreamDestroy(stream1));
    CUDA_CHECK(cudaStreamDestroy(stream2));

    return 0;
}
