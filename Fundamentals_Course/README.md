# CUDA Practice Repository

This repository serves as a practical playground for learning and experimenting with CUDA, based on the CUDA Fundamentals Course.

## Section 1: Fundamentals of CUDA

### GPU-accelerated vs. CPU-only applications

CUDA enables parallel processing on GPUs, offering significant speedups for certain types of computations compared to traditional CPU-only approaches. Understanding the differences between GPU-accelerated and CPU-only applications is crucial for optimizing performance.

### CUDA Kernel Execution

CUDA Kernels are parallel functions that run on the GPU. Exploring CUDA Kernel Execution is fundamental to harnessing the power of parallelism in GPU programming.

### Parallel Memory Access

Efficient memory access is key to maximizing GPU performance. This section delves into techniques for parallel memory access, optimizing data movement between CPU and GPU.

## Section 2: Advanced CUDA Concepts

### Streaming

Streaming is a technique for overlapping data transfers and computations, enhancing overall throughput. Explore how streaming can improve the efficiency of your CUDA applications.

### Non-Unified Memory

Understanding Non-Unified Memory is essential for managing data across CPU and GPU memory spaces separately. Learn how to leverage Non-Unified Memory for optimal resource utilization.

### cudaMemcpyAsync

`cudaMemcpyAsync` is an asynchronous memory copy function that allows concurrent data transfers and computation. Mastering `cudaMemcpyAsync` is crucial for achieving high-performance data movement.

## Section 3: Architecture Insights

### Streaming Multiprocessors

Streaming Multiprocessors (SMs) are the building blocks of the GPU architecture. Gain insights into how SMs execute parallel threads and learn strategies for maximizing their utilization.

### Unified Memory Behavior

Unified Memory simplifies memory management by providing a single, unified address space for CPU and GPU. Explore the behavior and best practices associated with Unified Memory in CUDA applications.

# CUDA Glossary

1. GPU (Graphics Processing Unit)

   A specialized processor designed for rendering graphics and performing parallel computations. In CUDA, GPUs are utilized for general-purpose parallel computing.

2. CUDA (Compute Unified Device Architecture)

   NVIDIA's parallel computing platform and programming model that enables developers to use NVIDIA GPUs for general-purpose processing.

3. Kernel

   A CUDA function that runs in parallel on the GPU. Kernels are a fundamental concept in CUDA programming and are executed by multiple threads.

   - Execution context: Special arguments given to CUDA kernels when launched using the <<<…>>> syntax. It defines the number of blocks in the grid, as well as the number of threads in each block.

4. Thread

   The smallest unit of execution in CUDA. Threads run in parallel on the GPU and collectively perform the work specified by a CUDA Kernel.

   - threadIdx.x: CUDA variable available inside executing kernel that gives the index the thread within the block

5. Block

   A group of threads that execute together on the GPU. Blocks are organized into a grid, and threads within a block can cooperate through shared memory.

   - blockDim.x: CUDA variable available inside executing kernel that gives the number of threads in the thread’s block
   - blockIdx.x: CUDA variable available inside executing kernel that gives the index the thread’s block within the grid

   - threadIdx.x + blockIdx.x \* blockDim.x: Common CUDA technique to map a thread to a data element

6. Grid

   A collection of blocks that execute a CUDA Kernel. The grid and block structure helps define the parallelism in CUDA applications.

   - gridDim.x: CUDA variable available inside executing kernel that gives the number of blocks in the grid

   - Grid-stride loop: A technique for assigning a thread more than one data element to work on when there are more elements than the number of threads in the grid. The stride is calculated by gridDim.x \* blockDim.x, which is the number of threads in the grid.

7. Shared Memory

   A region of memory shared among threads within the same block. It is faster than global memory and is used for communication and data sharing between threads.

8. Global Memory

   The main memory space accessible to both the CPU and GPU. Global memory is slower than shared memory but provides a larger storage capacity.

9. Warp

   A group of 32 parallel threads that execute in lockstep on a GPU. Warps are the basic units of execution on NVIDIA GPUs.

10. Warp Divergence

    A situation where threads within a warp take different execution paths, potentially leading to performance degradation. Minimizing warp divergence is crucial for optimal performance.

11. Streaming Multiprocessor (SM)

    The processing unit on a GPU responsible for executing CUDA threads. Understanding SM architecture is essential for optimizing GPU performance.

12. Asynchronous Execution

    The ability to overlap computation and data transfer operations. Asynchronous execution is crucial for maximizing GPU utilization.

    - cudaMallocManaged(): CUDA function to allocate memory accessible by both the CPU and GPUs. Memory allocated this way is called unified memory and is automatically migrated between the CPU and GPUs as needed.
    - cudaDeviceSynchronize(): CUDA function that will cause the CPU to wait until the GPU is finished working.

13. Unified Memory

    A memory management feature that allows a single memory space to be shared between the CPU and GPU, simplifying memory management in CUDA applications.

14. cudaMemcpy

    A function for copying data between the CPU and GPU. Understanding cudaMemcpy and its variants is essential for efficient data movement.

15. Compute Capability

    A version number assigned to a GPU architecture, indicating its features and capabilities. It is important to consider compute capability when optimizing CUDA code.

This glossary provides a starting point for understanding key terms in CUDA programming. As you delve deeper into CUDA, you may encounter additional terms and concepts that enhance your understanding of GPU computing.
