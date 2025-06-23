
//  Histogram Computation (CUDA)
// Develop a histogram computation using CUDA kernels .
//  Provide performance comparisons and rigorous correctness checks against single-threaded solutions.


#include <iostream>                              // For std::cout, std::endl, etc.
#include <cuda_runtime.h>                 // For CUDA API functions like cudaMalloc, cudaMemcpy, etc.
#include <chrono>                          // For computig  timing (CPU & GPU)
  
#define BIN_COUNT 256                  // Number of histogram bins
 
// CUDA kernel to compute histogram
__global__ void histogram_kernel(unsigned char* data, int* histo, int size) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;           //
    if (i < size) {
        atomicAdd(&histo[data[i]], 1);                // safely increments the corresponding histogram bin.
    }
}
 
void run_custom_histogram_test() {
    std::cout << "\n--- Histogram Test: Custom Data ---\n";
 
    // Custom input array for testing histogram
    int custom_data[] = {0, 1, 3, 2, 4, 6, 3, 8, 2, 4, 4, 8, 2, 3};
    int data_size = sizeof(custom_data) / sizeof(custom_data[0]);           // datasize holds the number of elemets in array
 
    // Allocate memory for input data
    unsigned char* data = new unsigned char[data_size];
    for (int i = 0; i < data_size; ++i)
        data[i] = static_cast<unsigned char>(custom_data[i]);
 
    // --- CPU Histogram ---
    int hist_cpu[BIN_COUNT] = {0};
    auto start_cpu = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < data_size; ++i)
        hist_cpu[data[i]]++;                                               //For each value in data, increment the corresponding bin in hist_cpu.
    auto end_cpu = std::chrono::high_resolution_clock::now();
    double cpu_time = std::chrono::duration<double>(end_cpu - start_cpu).count();
 
    // --- GPU Setup ---
    unsigned char* d_data;
    int* d_hist;
    cudaMalloc(&d_data, data_size);
    cudaMalloc(&d_hist, BIN_COUNT * sizeof(int));
    cudaMemcpy(d_data, data, data_size, cudaMemcpyHostToDevice);
    cudaMemset(d_hist, 0, BIN_COUNT * sizeof(int));          
 
    // Launch GPU kernel
    int threads = 256;
    int blocks = (data_size + threads - 1) / threads;            //ensures that there are enough threads launched to handle every element up to N.
 
    cudaDeviceSynchronize();                                                            //  the host (CPU) thread to block until all previously issued work on the device (GPU)
    auto start_gpu = std::chrono::high_resolution_clock::now();
    histogram_kernel<<<blocks, threads>>>(d_data, d_hist, data_size);
    cudaDeviceSynchronize();                                                               
    auto end_gpu = std::chrono::high_resolution_clock::now();
    double gpu_time = std::chrono::duration<double>(end_gpu - start_gpu).count();
 
    // --- Copy Results from GPU ---
    int hist_gpu[BIN_COUNT];
    cudaMemcpy(hist_gpu, d_hist, BIN_COUNT * sizeof(int), cudaMemcpyDeviceToHost);
 
    // --- Compare CPU and GPU Results ---
    bool match = true;
    for (int i = 0; i < BIN_COUNT; ++i) {
        if (hist_cpu[i] != hist_gpu[i]) {
            match = false;
            std::cout << "Mismatch at bin " << i << ": CPU = " << hist_cpu[i]
<< ", GPU = " << hist_gpu[i] << "\n";
        }
    }
 
    // Print results
    std::cout << "CPU Time: " << cpu_time << " s\n";
    std::cout << "GPU Time: " << gpu_time << " s\n";
    std::cout << (match ? "Histogram matches.\n" : "Histogram mismatch!\n");
 
    std::cout << "\nHistogram output:\n";
    for (int i = 0; i < 16; ++i) {
        if (hist_cpu[i] > 0)
            std::cout << "Bin " << i << " => " << hist_cpu[i] << "\n";
    }
 
    // Free memory
    delete[] data;
    cudaFree(d_data);
    cudaFree(d_hist);
}
 
int main() {
    run_custom_histogram_test();
    return 0;
}
