#  CUDA Vector Addition (C++17)

This project demonstrates how to perform **vector addition** using both:
- A **sequential CPU implementation**
- A **parallel CUDA (GPU) implementation**

# What is a Vector?
A vector in this context is a one-dimensional dynamic array that stores a list of numbers or elements in a contiguous block of memory.

In C++, std::vector is a standard template library (STL) container.

# What is CUDA?
CUDA (Compute Unified Device Architecture) is a parallel computing platform and application programming interface (API) created by NVIDIA. It allows developers to use the GPU for general-purpose processing



#  Features
- Adds two large vectors (`float` arrays) element-wise
- Runs the computation on both CPU and GPU
- Measures and compares performance
- Checks result correctness with a simple tolerance threshold
- Uses modern C++17 features


#  File Structure


vector_add/
├── vectoraddition.cu     # Main C++17 CUDA source file
├── README.md         # This file



##  How It Works

This project compares two methods of adding vectors: *on the CPU (sequential)*and *on the GPU using CUDA (parallel)*.

# CPU :

1. *Memory Allocation*:
   - Allocate vectors `A`, `B`, and `C_CPU` using `std::vector`.

2. *Initialization*:
   - Fill `A` and `B` with values like `A[i] = i`, `B[i] = 2 * i`.

3. *Computation*:
   - Loop through each index `i` and add:
     
     C_CPU[i] = A[i] + B[i];
     ```

4. *Timing*:
   - Uses `std::chrono` to record time taken for CPU computation.


#  GPU (CUDA) :


   1. **Host Memory Setup**:
   - Allocate host vectors `h_A`, `h_B`, and `h_C_GPU`.

2. **Device Memory Allocation**:
   - Allocate memory on the GPU using `cudaMalloc()`:
     
     float *d_A, *d_B, *d_C;
     cudaMalloc(&d_A, size);
     ```

3. **Copy Data to Device**:
   - Copy input vectors from host (CPU) to device (GPU) with `cudaMemcpy()`.

4. **Launch CUDA Kernel**:
   - Define how many threads and blocks are needed:
     ```cpp
     int threadsPerBlock = 256;
     int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
     ```

5. **CUDA Kernel Execution**:
   - GPU executes the kernel in parallel:
     ```cpp
     __global__ void vectorAddCUDA(...) {
         int i = blockIdx.x * blockDim.x + threadIdx.x;
         if (i < N) C[i] = A[i] + B[i];                                              
     }                                                                           
 i=(block index)×(threads per block)+(thread index within the block)
     ```

6. **Copy Results Back to Host**:
- Use `cudaMemcpy()` to transfer results from GPU back to CPU.


# How to run 

1.Go to 
https://leetgpu.com/playground

2.Run it 





# Sample output:
CPU Time: 0.000473246 seconds
GPU Time: 0.00172604 seconds
Results match: Yes 
