
//  Histogram Computation (CUDA)
// Develop a histogram computation using CUDA kernels .
//  Provide performance comparisons and rigorous correctness checks against single-threaded solutions.





#include <iostream>
#include <cstdlib>       
#include <ctime>               // fofr time libraries                 
#include <chrono>                // for computing time of cpu & gpu 
#include <cuda_runtime.h>              // CUDA Runtime API header i

#define BIN_COUNT 256           /// Number of histogram bins

// CUDA kernel
__global__ void histogram_kernel(unsigned char* data, int* histo, int size) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;   //i=(block index)×(threads per block)+(thread index within block) example formula
    if (i < size) {
        atomicAdd(&histo[data[i]], 1);    //  thread-safe increment of a specific element in the histogram array.

    }
}

void run_histogram_test(int data_size) {
    std::cout << "\n--- Histogram Test: Data Size = " << data_size << " ---\n";

    // Generate random input
    unsigned char* data = new unsigned char[data_size];                         //allocates memory dynamically for an array (creates a pointer named data that points to this newly allocated memory.)
    for (int i = 0; i < data_size; ++i)
        data[i] = rand() % BIN_COUNT;         // assigns a random value to the ith element of the array data.

    // CPU histogram
    int hist_cpu[BIN_COUNT] = {0};              //declares and initializes an array called cpu with a size of BIN_COUNT, setting all its elements to zero.


    auto start_cpu = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < data_size; ++i)
        hist_cpu[data[i]]++;
    auto end_cpu = std::chrono::high_resolution_clock::now();
    double cpu_time = std::chrono::duration<double>(end_cpu - start_cpu).count();

    // GPU memory allocation
    unsigned char* d_data;    //declares a pointer named d_data
    int* d_hist;
    cudaMalloc(&d_data, data_size);                // memory allocation
    cudaMalloc(&d_hist, BIN_COUNT * sizeof(int));
    cudaMemcpy(d_data, data, data_size, cudaMemcpyHostToDevice);             // memory copying
    cudaMemset(d_hist, 0, BIN_COUNT * sizeof(int));           //sets a block of GPU memory starting at d_hist to zero.

    // Launch kernel
    int threads = 256;
    int blocks = (data_size + threads - 1) / threads;    //The formula ensures that all data elements are assigned to threads for parallel processing.

    cudaDeviceSynchronize();        //It blocks the CPU thread until all previously submitted CUDA tasks on the device are completed.
    auto start_gpu = std::chrono::high_resolution_clock::now();
    histogram_kernel<<<blocks, threads>>>(d_data, d_hist, data_size);
    cudaDeviceSynchronize();
    auto end_gpu = std::chrono::high_resolution_clock::now();
    double gpu_time = std::chrono::duration<double>(end_gpu - start_gpu).count();

    // Copy back results
    int hist_gpu[BIN_COUNT];
    cudaMemcpy(hist_gpu, d_hist, BIN_COUNT * sizeof(int), cudaMemcpyDeviceToHost);

    // Compare results
    bool match = true;
    for (int i = 0; i < BIN_COUNT; ++i) {
        if (hist_cpu[i] != hist_gpu[i]) {
            match = false;
            std::cout << "Mismatch at bin " << i << ": CPU = " << hist_cpu[i]
                      << ", GPU = " << hist_gpu[i] << "\n";
            break;
        }
    }

    std::cout << "CPU Time: " << cpu_time << " s\n";
    std::cout << "GPU Time: " << gpu_time << " s\n";
    std::cout << (match ? " Histogram matches.\n" : " Histogram mismatch!\n");

    delete[] data;
    cudaFree(d_data);
    cudaFree(d_hist);
}

int main() {
    srand(static_cast<unsigned>(time(0)));                 // the random number generator with the current time to produce different pseudo-random sequences each run.

    // Test with 3 sizes only
    int sizes[] = {
        1 << 10,   // 1K
        1 << 16,   // 64K
        1 << 20    // 1M
    };

    for (int size : sizes) {
        run_histogram_test(size);
    }

    return 0;
}
