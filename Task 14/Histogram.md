
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

 # How it works
1.Histogram Calculation on CPU
'''Uses a loop to count occurrences of values in a 256-bin histogram.
Sequentially increments bin counts for each data element.'''

2.Histogram Calculation on GPU (CUDA)
'''Uses a CUDA kernel where each thread increments a histogram bin using atomicAdd.
Parallel processing allows faster computation compared to the CPU.'''
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
blocks = cell(dataSize / 256.0)
```

---

# CPU vs GPU Comparison

- **CPU**: Loops over the input data and increments the count of the appropriate bin.
- **GPU**: Launches many threads in parallel, each handling one input value.


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
--- Histogram Test: Custom Data ---
CPU Time: 1.1e-07 s = o.oooooo1
GPU Time: 0.000808953 s
Histogram matches.
 
Histogram output:
Bin 0 => 1
Bin 1 => 1
Bin 2 => 3
Bin 3 => 3
Bin 4 => 3
Bin 6 => 1
Bin 8 => 2
