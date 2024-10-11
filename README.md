# Reduction

-------------------

## Introduction

This program aims to implement and optimize parallel reduction operations in CUDA.  It is based on NVIDIA's PPT "Optimizing Parallel Reduction in CUDA" and fully implements the five-step optimization strategy proposed in the document.  After each optimization step, we verify the correctness of the optimized results by comparing them with the original reduction results.  On the NVIDIA GeForce RTX 4090 and NVIDIA GeForce RTX 3060 Laptop platforms, we tested the execution time of the program after each optimization step and compared it with the reduction summation functions of the Thrust and CUB libraries.  The test results show that the reduction operation after the five-step optimization is slightly faster than thrust::reduce function.

## Installation
#### Step 1 : Assert NVIDIA GPU and CUDA is available
To execute the Reduction program, verify that your system is equipped with an NVIDIA GPU and has the CUDA installed.

#### Step 2: Verify Your GPU's Compute Capability
The `nvcc` compilation command in the Makefile requires the compute capability of your GPU device (device 0). If you are unsure about the compute capability of your GPU device (device 0), you can run the script `GPU_information.sh` (execute the command `sh GPU_information.sh` in the terminal) to obtain this information.

#### Step 3: Confirm the Correct Makefile Settings
You can replace `arch=compute_86,code=sm_86` in the compilation command of the Makefile with the compute capability of your GPU device (device 0). Once this is done, you can proceed to execute the Makefile to compile the Reduction program.

## Build flags
`PANGULU_FLAGS` influences build behaviors. You can edit `PANGULU_FLAGS` in `make.inc` to implement different features of PanguLU. Here are available flags :

#### Decide if or not using GPU.
Use `-DGPU_OPEN` to use GPU, vice versa. Please notice that using this flag is not the only thing to do if you want to use GPU. Please check Step 8 in the Installation part.

#### Decide the value type of matrix and vector entries.
Use `-DCALCULATE_TYPE_R64` (double real) or `-DCALCULATE_TYPE_CR64` (double complex) or `-DCALCULATE_TYPE_R32` (float real) or `-DCALCULATE_TYPE_CR32` (float complex).

#### Decide if or not using MC64 reordering algorithm.
Use `-DPANGULU_MC64` to enable MC64 algorithm. Please notice that MC64 is not supported when matrix entries are complex numbers. If complex values are selected and `-DPANGULU_MC64` flag is used, MC64 would not enable.

#### Decide if or not using METIS reordering tool.
Use `-DMETIS` to enable METIS.

#### Decide log level.
Please select zero or one of these flags : `-DPANGULU_LOG_INFO`, `-DPANGULU_LOG_WARNING` or `-DPANGULU_LOG_ERROR`. Log level "INFO" prints all messages to standard output (including warnings and errors). Log level "WANRING" only prints warnings and errors. Log level "ERROR" only prints fatal errors causing PanguLU to terminate abnormally.

#### Decide core binding strategy.
Hyper-threading is not recommended. If you can't turn off the hyper-threading and each core of your CPU has 2 threads, using `-DHT_IS_OPEN`
may reaps performance gain.

## Executing the example code of PanguLU
The test routines are placed in the `examples` directory. The routine in `examples/example.c` firstly call `pangulu_gstrf()` to perform LU factorization, and then call `pangulu_gstrs()` to solve linear equation.
#### run command

> **mpirun -np process_count ./pangulu_example.elf -nb block_size -f path_to_mtx**
 
process_count : MPI process number to launch PanguLU;

block_size : Rank of each non-zero block;

path_to_mtx : The matrix name in mtx format.

You can also use the run.sh, for example:

> **bash run path_to_mtx block_size process_count**

#### test sample

> **mpirun -np 6 ./pangulu_example.elf -nb 4 -f Trefethen_20b.mtx**

or use the run.sh:
> **bash run.sh Trefethen_20b.mtx 4 6**


In this example, 6 processes are used to test, the block_size is 4, matrix name is Trefethen_20b.mtx.


## Release versions

#### <p align='left'>Version 1.0.0 (Oct. 19, 2021) </p>

* Used a rule-based 2D LU factorisation scheduling strategy.
* Used Sparse BLAS for floating point calculations on GPUs.
* Added the pre-processing phase.
* Added the numeric factorisation phase.
* Added the triangular solve phase.

## Reference

* [1] Xu Fu, Bingbin Zhang, Tengcheng Wang, Wenhao Li, Yuechen Lu, Enxin Yi, Jianqi Zhao, Xiaohan Geng, Fangying Li, Jingwen Zhang, Zhou Jin, Weifeng Liu. PanguLU: A Scalable Regular Two-Dimensional Block-Cyclic Sparse Direct Solver on Distributed Heterogeneous Systems. 36th ACM/IEEE International Conference for High Performance Computing, Networking, Storage, and Analysis (SC ’23). 2023.
