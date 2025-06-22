
# CUDA Histogram Computation with CPU Comparison

# Project Description

This project implements histogram computation using both:
- A **CUDA-based GPU kernel**
- A **single-threaded CPU loop**

It compares **performance** and checks the **correctness** of GPU results by matching them against the CPU output. The implementation is written in **C++17** with CUDA.

---

#What is a Histogram?

A histogram is a data structure that counts how many times each unique value (or value range) occurs in a dataset.

For example, if the data contains values like `[2, 2, 1, 0, 2, 0]` and the number of bins is 3, then:
```
Bin 0 → 2 counts  
Bin 1 → 1 count  
Bin 2 → 3 counts
```

---

#  What is CUDA?

CUDA (Compute Unified Device Architecture)** is a parallel computing platform developed by NVIDIA that allows developers to write software to run on the GPU (Graphics Processing Unit).

CUDA allows massively parallel computations by running thousands of lightweight threads concurrently, making it well-suited for tasks like histogram computation.

---

# How the CUDA Kernel Works

Each thread processes one element of the input data and updates the corresponding bin using:

```cpp
atomicAdd(&histo[data[i]], 1);
```

This atomic operation ensures that multiple threads updating the same bin do not create race conditions.

---

# Grid and Block Size Formula

The kernel is launched with a formula that determines how many threads and blocks are needed:

```cpp
int threadsPerBlock = 256;
int blocks = (dataSize + threadsPerBlock - 1) / threadsPerBlock;
```

# Why this formula?

This ensures that if `dataSize` isn't an exact multiple of 256, the last partial block still processes the remaining data. It is equivalent to:
```
blocks = ceil(dataSize / 256.0)
```

---

# CPU vs GPU Comparison

- **CPU**: Loops over the input data and increments the count of the appropriate bin.
- **GPU**: Launches many threads in parallel, each handling one input value.

The program runs tests on **3 data sizes**:
- 1K (1024 elements)
- 64K (65536 elements)
- 1M (1048576 elements)

For each test, it reports:
- Execution time on CPU and GPU
- Whether the histograms match

---





# How to run 

1.Go to 
https://leetgpu.com/playground

2.Run it 
---



# sample output:
--- Histogram Test: Data Size = 1024 ---
CPU Time: 2.918e-06 s
GPU Time: 0.00172833 s
✅ Histogram matches.

--- Histogram Test: Data Size = 65536 ---
CPU Time: 0.00017625 s
GPU Time: 0.000820406 s
✅ Histogram matches.

--- Histogram Test: Data Size = 1048576 ---
CPU Time: 0.00166255 s
GPU Time: 0.202779 s
✅ Histogram matches.