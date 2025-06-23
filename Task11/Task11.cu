
// Task 11//

// CUDA Parallel Vector Addition 
 //Implement parallel vector addition using CUDA.
 // Compare performance, correctness, and scalability against sequential C++ implementations.



#include <iostream>
#include <vector>
#include <chrono>               // for computing time of cpu & gpu 
#include <cmath>                // for mathematical functions
#include <cuda_runtime.h>

using namespace std;

int N = 1 << 16;                                // 65536 elements

// CUDA Kernel: Vector addition on GPU
__global__ void vectorAddCUDA(const float* A, const float* B, float* C, int n) {                     
    int i = blockIdx.x * blockDim.x + threadIdx.x;                                         //i=(block index)×(threads per block)+(thread index within the block)       (i is the global index)                                             
    if (i < n){
        C[i] = A[i] + B[i];
}
}

// Sequential vector addition on CPU
void vectorAddCPU(const vector<float>& A, const vector<float>& B, vector<float>& C) {               
    for (int i = 0; i < N; ++i) {
        C[i] = A[i] + B[i];
    }
}

// Verify CPU and GPU results match
bool verify(const vector<float>& ref, const vector<float>& test) {
    for (int i = 0; i < N; ++i) {
        if (fabs(ref[i] - test[i]) > 1e-5f) {                                                               //Checks if the absolute difference between ref[i] and test[i] exceeds a tiny threshold 1e-5
            cerr << "Mismatch at index " << i << ": " << ref[i] << " != " << test[i] << "\n";            // reports index if it mismatches
            return false;
        }
    }
    return true;
}

int main() {
    vector<float> h_A(N), h_B(N), h_C_CPU(N), h_C_GPU(N);    // h refers to the host  memory

    // Initialize vectors
    for (int i = 0; i < N; ++i) {
        h_A[i] = static_cast<float>(i);                      
        h_B[i] = static_cast<float>(2 * i);     //type casting 
    }

    // --- CPU computation ---
    auto startCPU = chrono::high_resolution_clock::now();           
    vectorAddCPU(h_A, h_B, h_C_CPU);
    auto endCPU = chrono::high_resolution_clock::now();
    chrono::duration<double> cpuTime = endCPU - startCPU;
    cout << "CPU Time: " << cpuTime.count() << " seconds\n";

    // --- GPU computation ---
    float *d_A, *d_B, *d_C;                               // d refers to deive memomry
    cudaMalloc(&d_A, N * sizeof(float));
    cudaMalloc(&d_B, N * sizeof(float));
    cudaMalloc(&d_C, N * sizeof(float));                     // memory alloc

    cudaMemcpy(d_A, h_A.data(), N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), N * sizeof(float), cudaMemcpyHostToDevice);           // memory copy

    int threadsPerBlock = 256;                                                              
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    cudaEvent_t startGPU, stopGPU;                                                      
    cudaEventCreate(&startGPU);
    cudaEventCreate(&stopGPU);

    cudaEventRecord(startGPU);
    vectorAddCUDA<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
    cudaEventRecord(stopGPU);

    cudaMemcpy(h_C_GPU.data(), d_C, N * sizeof(float), cudaMemcpyDeviceToHost);

    cudaEventSynchronize(stopGPU);                                   // wait for the event to complete
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, startGPU, stopGPU);              
    cout << "GPU Time: " << milliseconds / 1000.0 << " seconds\n";             // how long cuda takes time to execute
  
    // --- Verify correctness ---
    bool correct = verify(h_C_CPU, h_C_GPU);
    cout << "Results match: " << (correct ? "Yes " : "No ") << endl;

    // Cleanup
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaEventDestroy(startGPU);
    cudaEventDestroy(stopGPU);

    return 0;
}
