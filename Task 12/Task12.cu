
// Task 12//


// CUDA Parallel Matrix Transpose
// Implement a matrix transpose operation using CUDA kernels.//
// Validate results rigorously against CPU-based matrix transpose  //
 





#include <iostream>
#include <cuda_runtime.h>       // CUDA Runtime API header     
#include <chrono>               // for computing time of cpu & gpu 
#include <cmath>                 // for mathematical functions


#define N  8                                       // Matrix size
  
//cuda kernel
__global__ void matrixTransposeKernel(float* d_out, const float* d_in, int width) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < width && col < width) {
        d_out[col * width + row] = d_in[row * width + col];   
    }
}

void cpuMatrixTranspose(float* out, const float* in, int width) {
    for (int row = 0; row < width; ++row)                                              // Transposing
        for (int col = 0; col < width; ++col)
            out[col * width + row] = in[row * width + col];
}

bool validate(const float* a, const float* b, int size) {
    for (int i = 0; i < size; ++i)
        if (std::fabs(a[i] - b[i]) > 1e-5)
            return false;
    return true;
}

void printMatrix(const float* mat, int width) {
    for (int i = 0; i < width; ++i) {
        for (int j = 0; j < width; ++j)
            std::cout << mat[i * width + j] << "\t";
        std::cout << "\n";
    }
    std::cout << "\n";
}

int main() {
    const int SIZE = N * N * sizeof(float);                     // total memory size in bytes needed to store an N x N

    float h_in[N * N], h_out_gpu[N * N], h_out_cpu[N * N];

    // Initialize matrix
    for (int i = 0; i < N * N; ++i)
        h_in[i] = static_cast<float>(i + 1);

    // CPU Timing
    auto cpu_start = std::chrono::high_resolution_clock::now();
    cpuMatrixTranspose(h_out_cpu, h_in, N);
    auto cpu_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::micro> cpu_duration = cpu_end - cpu_start;

    // GPU Timing using cudaEvent
    float *d_in, *d_out;
    cudaMalloc(&d_in, SIZE);
    cudaMalloc(&d_out, SIZE);
    cudaMemcpy(d_in, h_in, SIZE, cudaMemcpyHostToDevice);

    dim3 threads(8, 8);                     //A CUDA-specific type that defines a 3-dimensional vector, used to specify the number of threads and blocks in each dimension.
    dim3 blocks(1, 1);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    matrixTransposeKernel<<<blocks, threads>>>(d_out, d_in, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);                      //is a CUDA runtime API call used in CUDA programming to block the CPU thread until the specified CUDA event (stop) has completed.

    float gpu_time_ms = 0;
    cudaEventElapsedTime(&gpu_time_ms, start, stop);                     //CUDA programming to measure the elapsed time between two CUDA events

    cudaMemcpy(h_out_gpu, d_out, SIZE, cudaMemcpyDeviceToHost);

    // Output
    std::cout << "Original Matrix:\n";
    printMatrix(h_in, N);

    std::cout << "CPU Transpose:\n";
    printMatrix(h_out_cpu, N);

    std::cout << "GPU Transpose:\n";
    printMatrix(h_out_gpu, N);

    std::cout << (validate(h_out_cpu, h_out_gpu, N * N) ? " Transpose Correct\n" : " Incorrect Transpose\n");

    std::cout << "CPU Time: " << cpu_duration.count() << " microseconds\n";
    std::cout << "GPU Time: " << gpu_time_ms * 1000.0 << " microseconds\n";  // ms to µs

    // Cleanup
    cudaFree(d_in);
    cudaFree(d_out);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
