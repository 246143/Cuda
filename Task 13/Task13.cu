//Task 13: Parallel Prime Number Finder (CUDA/Multithreaded) 
// Implement a CUDA or multi-threaded prime number finder up to 1 million.
// Benchmark and provide clear performance comparisons against sequential implementations.






#include <iostream>
#include <cmath>   //for mathematical functions
#include <cuda.h>           
#include <chrono>         // for computing time of cpu & gpu 

#define N 1000000  //                                                                     Find primes up to 1 million

// =================== CUDA Kernel =================== //
__global__ void findPrimesCUDA(bool* d_primes) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x + 2;                              // Skip 0 and 1
    if (idx > N) return;

    bool prime = true;
    for (int i = 2; i <= sqrtf((float)idx); ++i) {   // it is To test if idx is divisible by any number from 2 up to the square root of idx.
        if (idx % i == 0) {
            prime = false;
            break;
        }
    }
    d_primes[idx] = prime;       //writes the value of prime into the device array d_primes at index idx.
}

// =================== CPU Prime Finder =================== //
void findPrimesCPU(bool* primes) {
    for (int i = 2; i <= N; ++i) {
        primes[i] = true;
        for (int j = 2; j * j <= i; ++j) {
            if (i % j == 0) {
                primes[i] = false;
                break;
            }
        }
    }
}

// =================== Count Prime Numbers =================== //
int countPrimes(bool* primes) {     // To count how many entries in the array indicate prime numbers.

    int count = 0;
    for (int i = 2; i <= N; ++i)
        if (primes[i]) count++;                  
    return count;
}

// =================== MAIN =================== //
int main() {
    // Allocate memory for CPU results
    bool* primesCPU = new bool[N + 1];      //Dynamically allocates an array of boolean values with size N + 1.
    bool* primesGPU = new bool[N + 1];

    // -------- CPU Prime Finder -------- //
    auto startCPU = std::chrono::high_resolution_clock::now();
    findPrimesCPU(primesCPU);                               // which numbers are prime
    auto endCPU = std::chrono::high_resolution_clock::now();
    double timeCPU = std::chrono::duration<double, std::milli>(endCPU - startCPU).count();    //milliseconds elapsed between startCPU and endCPU
    int countCPU = countPrimes(primesCPU);
    std::cout << "[CPU] Found " << countCPU << " primes in " << timeCPU << " ms\n";

    // -------- GPU Prime Finder -------- //
    bool* d_primes;
    cudaMalloc(&d_primes, (N + 1) * sizeof(bool));     // allocates memory on the GPU to store N+1 boolean values.]
    cudaMemset(d_primes, 0, (N + 1) * sizeof(bool));     //initialize or reset memory on the GPU.


 
    int blockSize = 256;
    int gridSize = (N - 1 + blockSize - 1) / blockSize;   //  ensures that there are enough threads launched to handle every element up to N.

    cudaDeviceSynchronize();                                      // the host (CPU) thread to block until all previously issued work on the device (GPU)
    auto startGPU = std::chrono::high_resolution_clock::now();
    findPrimesCUDA<<<gridSize, blockSize>>>(d_primes);
    cudaDeviceSynchronize();
    auto endGPU = std::chrono::high_resolution_clock::now();

    double timeGPU = std::chrono::duration<double, std::milli>(endGPU - startGPU).count();

    cudaMemcpy(primesGPU, d_primes, (N + 1) * sizeof(bool), cudaMemcpyDeviceToHost);
    int countGPU = countPrimes(primesGPU);

    std::cout << "[GPU] Found " << countGPU << " primes in " << timeGPU << " ms\n";
    std::cout << "Speedup: " << (timeCPU / timeGPU) << "x\n";

    // -------- Clean Up -------- //
    cudaFree(d_primes);
    delete[] primesCPU;
    delete[] primesGPU;

    return 0;
}
