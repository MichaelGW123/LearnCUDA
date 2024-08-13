#include <iostream>
#include <vector>
#include <chrono>

void vectorAdd(const std::vector<float> &A, const std::vector<float> &B, std::vector<float> &C, int N)
{
    for (int i = 0; i < N; ++i)
    {
        C[i] = A[i] + B[i];
    }
}

int main()
{
    int N = 1 << 28; // 1M elements
    std::vector<float> A(N, 1.0f);
    std::vector<float> B(N, 2.0f);
    std::vector<float> C(N);

    auto start = std::chrono::high_resolution_clock::now();
    vectorAdd(A, B, C, N);
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<float, std::milli> duration = end - start;
    std::cout << "Time taken: " << duration.count() << " ms" << std::endl;

    return 0;
}
