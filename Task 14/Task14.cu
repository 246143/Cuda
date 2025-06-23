#include <iostream>              // For std::cout
#include <cuda_runtime.h>        // CUDA API
#include <chrono>                // Timing
#include <cstdlib>               // rand(), srand()
#include <ctime>                 // time()

#define BIN_COUNT 256            // Number of histogram bins

// ---------------- CUDA Kernel ---------------- //
__global__ void histogram_kernel(unsigned char* data, int* histo, int size) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < size) {
        atomicAdd(&histo[data[i]], 1);
    }
}

// --------- Histogram Test for Input Size --------- //
void run_histogram_test(int data_size) {
    std::cout << "\n--- Histogram Test: Data Size = " << data_size << " ---\n";

    // Generate random input data
    unsigned char* data = new unsigned char[data_size];
    for (int i = 0; i < data_size; ++i)
        data[i] = rand() % BIN_COUNT;

    // ---------- CPU Histogram ---------- //
    int hist_cpu[BIN_COUNT] = {0};
    auto start_cpu = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < data_size; ++i)
        hist_cpu[data[i]]++;
    auto end_cpu = std::chrono::high_resolution_clock::now();
    double cpu_time = std::chrono::duration<double>(end_cpu - start_cpu).count();

    // ---------- GPU Setup ---------- //
    unsigned char* d_data;
    int* d_hist;
    cudaMalloc(&d_data, data_size);
    cudaMalloc(&d_hist, BIN_COUNT * sizeof(int));
    cudaMemcpy(d_data, data, data_size, cudaMemcpyHostToDevice);
    cudaMemset(d_hist, 0, BIN_COUNT * sizeof(int));

    // ---------- Launch Kernel ---------- //
    int threads = 256;
    int blocks = (data_size + threads - 1) / threads;

    cudaDeviceSynchronize();
    auto start_gpu = std::chrono::high_resolution_clock::now();
    histogram_kernel<<<blocks, threads>>>(d_data, d_hist, data_size);
    cudaDeviceSynchronize();
    auto end_gpu = std::chrono::high_resolution_clock::now();
    double gpu_time = std::chrono::duration<double>(end_gpu - start_gpu).count();

    // ---------- Copy Back Results ---------- //
    int hist_gpu[BIN_COUNT];
    cudaMemcpy(hist_gpu, d_hist, BIN_COUNT * sizeof(int), cudaMemcpyDeviceToHost);

    // ---------- Verify Results ---------- //
    bool match = true;
    for (int i = 0; i < BIN_COUNT; ++i) {
        if (hist_cpu[i] != hist_gpu[i]) {
            match = false;
            std::cout << "Mismatch at bin " << i << ": CPU = " << hist_cpu[i]
                      << ", GPU = " << hist_gpu[i] << "\n";
        }
    }

    std::cout << "CPU Time: " << cpu_time << " s\n";
    std::cout << "GPU Time: " << gpu_time << " s\n";
    std::cout << (match ? "✅ Histogram matches.\n" : "❌ Histogram mismatch!\n");

    // ---------- Show Non-zero Histogram Output ---------- //
    std::cout << "Sample Histogram (Non-zero bins):\n";
    for (int i = 0; i < 16; ++i) {
        if (hist_cpu[i] > 0)
            std::cout << "Bin " << i << " => " << hist_cpu[i] << "\n";
    }

    // ---------- Cleanup ---------- //
    delete[] data;
    cudaFree(d_data);
    cudaFree(d_hist);
}

// ----------------- Main Function ----------------- //
int main() {
    srand(time(0)); // Seed for random data

    int sizes[] = {
        1 << 10,   // 1K
        1 << 13,   // 8K
        1 << 16,   // 64K
        1 << 18,   // 256K
        1 << 20,   // 1M
    
    };

    for (int size : sizes) {
        run_histogram_test(size);
    }

    return 0;
}
