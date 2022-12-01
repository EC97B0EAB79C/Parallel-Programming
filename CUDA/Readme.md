# CUDA
Project to improve the performance of gravity calculation on CUDA.

## Improvements
These are improvements made to improve the performance.

1. [Thread](./Gravity/01Thread/kernel.cu): Begin parallelization with CUDA threads.
1. [Block](./Gravity/02Block/kernel.cu): Increase parallelization by including CUDA thread blocks.
1. [MultiStep](./Gravity/03MultiStep/kernel.cu): Divide kernel code into multi-steps to change the degree of parallelization according to calculation.
1. [Instruction](./Gravity/04Instruction/kernel.cu): Optimize instructions to increase performance.
1. [Shared Memory](./Gravity/05SharedMemory/kernel.cu): Used shared memory.
1. [Reduction](./Gravity/06Reduction/kernel.cu): Reduction with Warp Shuffle.

## Failed Attempts
List of failed attempts for increasing performance. 

1. [Dimension](./Gravity/99Failed/04Dimension/kernel.cu): Tried to increase performance by increasing parallelization on dimension(XYZ) 
1. [Shift](./Gravity/99Failed/04Shift/kernel.cu): Tried to increase performance by reducing multiplication with shift calculation.

## Runtime (Intel i7-8700k and NVIDIA Geforce RTX 2070)
2. 1875.917236ms
3. 1202.103394ms
4. 863.557068ms
5. 838.583557ms

## Disclaimer
**Please do not use programs in this repository *as-is* for purposes like assignments.**
