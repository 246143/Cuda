

# Parallel Prime Number Finder using CUDA and CPU (C++17)

This project finds **all prime numbers up to 1,000,000** using:

-  A **sequential CPU** implementation (C++17)
-  A **parallel GPU** implementation (CUDA)

It compares both for:
-  Correctness
-  Performance (timing)
-  Speedup (how much faster the GPU is)

---

#  Objective

- Find prime numbers from **2 to 1,000,000**
- Use **CUDA** to run primality checks in parallel
- Compare with a classic **CPU loop**
- Report number of primes found and runtime for each approach

---

# Key Concepts

# What Is a Prime Number?

A prime number is a natural number greater than 1 that has no positive divisors other than 1 and itself.
A number n is prime if it's not divisible by any number from 2 to √n.
Examples:
-  Prime: `2`, `3`, `5`, `7`, `11`
-  Not Prime: `4`, `6`, `9`, `12`

---

# What Is CUDA?

CUDA (Compute Unified Device Architecture) is a parallel computing platform developed by **NVIDIA** that allows running thousands of lightweight threads on the GPU.

With CUDA, we can:
- Perform tasks faster through **massive parallelism**
- Leverage GPU threads to check many numbers at once

---

# What Is a CUDA Kernel?

A **kernel** is a function that runs on the **GPU**, launched by the CPU, and executed by multiple **threads in parallel**.

In this project:
- Each thread checks whether a single number is prime.
- Threads run simultaneously, enabling massive speedup over a sequential CPU loop.

---

# Code Overview

 # How It Works
1.Prime Detection on CPU:

Uses a loop to check divisibility for each number up to N = 1,000,000.
Determines prime numbers sequentially.

2.Prime Detection on GPU (CUDA):

'''Uses a CUDA kernel where each thread evaluates a number’s primality in parallel.
Divides workload dynamically among GPU threads for efficiency.
Performance Comparison
Measures execution time on both CPU and GPU using std::chrono.
Computes speedup factor (CPU Time / GPU Time).
Ensures correctness by comparing results with an error tolerance'''

#  Performance Measurement

Execution time is measured using C++17 `std::chrono` for both CPU and GPU, and speedup is calculated:

```text
Speedup = CPU Time / GPU Time
```






# How to run 

1.Go to 
https://leetgpu.com/playground

2.Run it 
---

# Sample Output


[CPU] Found 78498 primes in 216.466 ms
[GPU] Found 78498 primes in 1338.8 ms
Speedup: 0.161687x
---

#  Memory Management

- CPU: `new[]` / `delete[]` for prime arrays  
- GPU: `cudaMalloc` / `cudaFree` for device memory  
- GPU results are copied back to host using `cudaMemcpy`

---

# Validation

The number of primes found on CPU and GPU are compared:

```text
if (countCPU == countGPU)
    ✅ Results match
else
    ❌ Results mismatch
```

Ensures correctness of the CUDA parallel result.

---

#  Cleanup

- Releases all heap and device memory to avoid memory leaks.
- Synchronization is used to ensure accurate timing and correctness.

---



---

# Project Structure

 File                 
 `primefider.cu`
 `prime README.md`        

-
