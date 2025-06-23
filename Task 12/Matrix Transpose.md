#  CUDA Parallel Matrix Transpose (8x8)

This project demonstrates how to implement a **parallel matrix transpose** using **CUDA** and validate the results against a CPU implementation. A fixed 8×8 matrix is transposed using both CPU and GPU, and performance (execution time) is compared.

---

# What is Parallel Matrix Transpose?

A **matrix transpose** is the operation of flipping a matrix over its diagonal — the row and column indices are swapped.

Given matrix A:
```
1 2 3
4 5 6
```

Its transpose Aᵗ is:
```
1 4
2 5
3 6
```

**Parallel transpose** distributes the transpose task across multiple processing units — like CPU threads or GPU cores — to speed up the operation, especially for large matrices.

---

# What is CUDA?

**CUDA (Compute Unified Device Architecture)** is a parallel computing platform and API created by NVIDIA. It allows developers to run C/C++ code on the **GPU (Graphics Processing Unit)**, taking advantage of its many cores for parallel execution.

## What is a CUDA Kernel?

A **CUDA kernel** is a function written in C/C++ and executed on the GPU. It runs **in parallel across many threads**, where each thread processes a portion of the data.

For example, in matrix transpose, each GPU thread handles one element swap from A[i][j] to A[j][i].

---

#  How It Works

- The code defines an 8×8 matrix initialized with values from 1 to 64.
- The matrix is transposed using:
  - A **CPU function** with nested loops
  - A **CUDA kernel** with 8×8 threads
- The output from both methods is printed.
- The results are compared for correctness.
- Execution time for CPU and GPU versions is displayed.

- Performance Comparison
Measures execution time on both CPU and GPU using std::chrono.
Computes speedup factor (CPU Time / GPU Time).
Ensures correctness by comparing results within an error tolerance.

Uses dynamic memory allocation (cudaMalloc, cudaMemcpy) for GPU processing.




# How to run 

1.Go to 
https://leetgpu.com/playground

2.Run it 

> 💡 **Note:** For small matrices (like 8x8), the CPU is faster due to GPU overhead.
> For larger matrices (e.g., 1024×1024), GPU becomes significantly faster.

---


# Sample output:
Original Matrix:
1       2       3       4       5       6       7       8
9       10      11      12      13      14      15      16
17      18      19      20      21      22      23      24
25      26      27      28      29      30      31      32
33      34      35      36      37      38      39      40
41      42      43      44      45      46      47      48
49      50      51      52      53      54      55      56
57      58      59      60      61      62      63      64

CPU Transpose:
1       9       17      25      33      41      49      57
2       10      18      26      34      42      50      58
3       11      19      27      35      43      51      59
4       12      20      28      36      44      52      60
5       13      21      29      37      45      53      61
6       14      22      30      38      46      54      62
7       15      23      31      39      47      55      63
8       16      24      32      40      48      56      64

GPU Transpose:
1       9       17      25      33      41      49      57
2       10      18      26      34      42      50      58
3       11      19      27      35      43      51      59
4       12      20      28      36      44      52      60
5       13      21      29      37      45      53      61
6       14      22      30      38      46      54      62
7       15      23      31      39      47      55      63
8       16      24      32      40      48      56      64

 Transpose Correct
CPU Time: 0.255 microseconds
GPU Time: 557.41 microseconds

---


