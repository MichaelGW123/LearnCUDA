#include <iostream>
#include <math.h>

#define gpuCheck(ans)                     \
  {                                       \
    gpuAssert((ans), __FILE__, __LINE__); \
  }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
  if (code != cudaSuccess)
  {
    fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
    if (abort)
      exit(code);
  }
}

// Kernel function to add the elements of two arrays
__global__ void add(int n, float *x, float *y)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;

  for (int i = index; i < n; i += stride)
    y[i] = x[i] + y[i];
}

int main(void)
{
  int N = 1 << 30; // Increased from 1 << 20 to 1 << 30 for larger dataset
  float *x, *y;

  // Allocate Unified Memory – accessible from CPU or GPU
  gpuCheck(cudaMallocManaged(&x, N * sizeof(float)));
  gpuCheck(cudaMallocManaged(&y, N * sizeof(float)));

  // initialize x and y arrays on the host
  for (int i = 0; i < N; i++)
  {
    x[i] = 1.0f;
    y[i] = 2.0f;
  }

  // Run kernel on N elements on the GPU
  int numThreadsPerBlock = 256;                                      // You can adjust this value based on your GPU's capabilities
  int numBlocks = (N + numThreadsPerBlock - 1) / numThreadsPerBlock; // Round up to ensure all elements are covered

  add<<<numBlocks, numThreadsPerBlock>>>(N, x, y);

  // Check for errors (all values should be 3.0f)
  gpuCheck(cudaPeekAtLastError());   // Check for kernel launch errors
  gpuCheck(cudaDeviceSynchronize()); // Synchronize device to ensure kernel execution is completed

  // Check for errors after kernel execution
  gpuCheck(cudaPeekAtLastError()); // Check for any errors during kernel execution

  // Free memory
  cudaFree(x);
  cudaFree(y);

  return 0;
}
