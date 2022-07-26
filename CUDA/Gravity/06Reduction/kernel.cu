﻿/*
CUDA code for making improvements by utilizing Shared memory
*/
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#include "../data_util/data_util.h"

#define N 10000
#define G 1
#define DT 1

#define kABlockDimX 16
#define kABlockDimY 16

#define kVBlockDimX 32
#define kVBlockDimY 2

__global__ void kernelAcceleration(double*, double*, double*, double*, double*);
__global__ void kernelVelocity(double*, double*, double*, double*, double*);


cudaError runKernel(double* m, double* a, double* v, double* pos) {


	dim3 blockDim;
	dim3 gridDim;
	float milisecondsStoM = 0;
	float milisecondsStoE = 0;


	cudaError_t cudaStatus;
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "Err: cudaSetDevice\n");
		goto Error;
	}

	size_t size;


	// device(CUDA) variable
	double* d_m;
	size = sizeof(double) * N;
	cudaStatus = cudaMalloc(&d_m, size);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "Err: cudaMalloc\n");
		goto Error;
	}

	double* d_a;
	size = sizeof(double) * N * N * 3;
	cudaStatus = cudaMalloc(&d_a, size);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "Err: cudaMalloc\n");
		goto Error;
	}

	double* d_v;
	size = sizeof(double) * N * 3;
	cudaStatus = cudaMalloc(&d_v, size);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "Err: cudaMalloc\n");
		goto Error;
	}

	double* d_pos;
	cudaStatus = cudaMalloc(&d_pos, size);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "Err: cudaMalloc\n");
		goto Error;
	}

	double* d_result;
	cudaStatus = cudaMalloc(&d_result, size);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "Err: cudaMalloc\n");
		goto Error;
	}


	// performance metric
	cudaEvent_t start, memcpy, end;
	cudaEventCreate(&start);
	cudaEventCreate(&memcpy);
	cudaEventCreate(&end);
	cudaEventRecord(start);	// record event


	// cudaMemcpy
	size = sizeof(double) * N;
	cudaStatus = cudaMemcpy(d_m, m, size, cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "Err: cudaMemcpy\n");
		goto Error;
	}

	size = sizeof(double) * N * 3;
	cudaStatus = cudaMemcpy(d_v, v, size, cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "Err: cudaMemcpy\n");
		goto Error;
	}

	cudaStatus = cudaMemcpy(d_pos, pos, size, cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "Err: cudaMemcpy\n");
		goto Error;
	}
	// record event
	cudaEventRecord(memcpy);


	// launch kernel
	for (int i = 0; i < 10; i++) {
		blockDim = dim3(kABlockDimX, kABlockDimY);
		gridDim = dim3((N + (blockDim.x - 1)) / blockDim.x, (N + (blockDim.y - 1)) / blockDim.y);
		kernelAcceleration << <gridDim, blockDim >> > (d_m, d_a, d_v, d_pos, d_result);
		cudaStatus = cudaDeviceSynchronize();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "Err: %dth iter Kernel\n", i);
			goto Error;
		}

		blockDim = dim3(kVBlockDimX, kVBlockDimY);
		gridDim = dim3((N + blockDim.y - 1) / blockDim.y);
		kernelVelocity << <gridDim, blockDim >> > (d_m, d_a, d_v, d_pos, d_result);
		cudaStatus = cudaDeviceSynchronize();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "Err: %dth iter Kernel\n", i);
			goto Error;
		}
	}


	// cudaMemcpy result from device
	cudaDeviceSynchronize();
	cudaStatus = cudaMemcpy(pos, d_pos, size, cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "Err: cudaMemcpy\n");
		goto Error;
	}
	cudaEventRecord(end);	// record event
	cudaDeviceSynchronize();

	// print performace metrics
	cudaEventElapsedTime(&milisecondsStoM, start, memcpy);
	fprintf(stdout, "Time from start to memcpy:\n\t%f\n", milisecondsStoM);
	cudaEventElapsedTime(&milisecondsStoE, start, end);
	fprintf(stdout, "Time from start to end:\n\t%f\n", milisecondsStoE);

Error:
	// free cuda memory
	cudaFree(d_m);
	cudaFree(d_a);
	cudaFree(d_v);
	cudaFree(d_pos);
	cudaFree(d_result);

	cudaDeviceReset();

	return cudaStatus;
}


int main() {
	size_t size;


	// host variable
	size = sizeof(double) * N;
	double* m = (double*)malloc(size);
	size = sizeof(double) * N * N * 3;
	double* a = (double*)malloc(size);
	size = sizeof(double) * N * 3;
	double* v = (double*)malloc(size);
	double* pos = (double*)malloc(size);


	// check variable
	if (
		m == NULL ||
		a == NULL ||
		v == NULL ||
		pos == NULL
		) {
		fprintf(stderr, "Err: Malloc Failed\n");
		return -1;
	}


	// set variable
	if (readDouble("../test/n10000/m.double", m) != N) {
		fprintf(stderr, "Err: Can not read m.double\n");
		return -1;
	}

	size = sizeof(double) * N * N * 3;
	memset(a, 0, size);

	size = sizeof(double) * N * 3;
	memset(v, 0, size);

	if (readDouble("../test/n10000/x.double", pos) != N) {
		fprintf(stderr, "Err: Can not read x.double\n");
		return -1;
	}
	if (readDouble("../test/n10000/y.double", pos + N) != N) {
		fprintf(stderr, "Err: Can not read y.double\n");
		return -1;
	}
	if (readDouble("../test/n10000/z.double", pos + N * 2) != N) {
		fprintf(stderr, "Err: Can not read z.double\n");
		return -1;
	}


	// run kernel
	cudaError_t cudaStatus;
	if ((cudaStatus = runKernel(m, a, v, pos)) != cudaSuccess) {
		fprintf(stderr, "Err: Kernal returned error code %d\n\t%s\n",
			cudaStatus,
			cudaGetErrorString(cudaStatus));
	}

	//printf("%lf", pos[9999]);
	// write results
	char writeFileX[] = "./cuda_x.double";
	writeDouble(writeFileX, pos, N);
	char writeFileY[] = "./cuda_y.double";
	writeDouble(writeFileY, pos + N, N);
	char writeFileZ[] = "./cuda_z.double";
	writeDouble(writeFileZ, pos + N * 2, N);


	// free memory
	free(m);
	free(a);
	free(v);
	free(pos);

	return 0;
}

// kernelAcceleration is a CUDA kernel that calculates the acceleration of each particle in the system.
// It takes the following arguments:
//   - m: an array of masses for each particle in the system
//   - a: an array of accelerations for each particle in the system
//   - v: an array of velocities for each particle in the system
//   - pos: an array of positions for each particle in the system
//   - result: an array where the results will be stored
__global__ void kernelAcceleration(double* m, double* a, double* v, double* pos, double* result) {
	// t_i and t_j are the indices of the current thread within its thread block
	int t_i = threadIdx.x;
	int t_j = threadIdx.y;
	// i and j are the global indices of the current thread
	// This is calculated by combining the thread block indices and the thread indices
	int i = blockDim.x * blockIdx.x + t_i;
	int j = blockDim.y * blockIdx.y + t_j;

	// If the global indices are out of bounds, return early
	if (i >= N || j >= N) return;

	// pos_i and pos_j are shared memory arrays used to store the positions of the current particle
	// and the particle that the current thread is interacting with, respectively.
	// They are declared with dimensions kABlockDimX * 3 and kABlockDimY * 3, respectively,
	// where kABlockDimX and kABlockDimY are constants that specify the dimensions of the thread block.
	__shared__ double pos_i[kABlockDimX * 3];
	__shared__ double pos_j[kABlockDimY * 3];
	
	// The positions of the current particle are copied from the global pos array into shared memory.
	// The positions are stored in separate elements for the x, y, and z coordinates.
	if(t_j < 3){
		pos_i[t_i + kABlockDimX * t_j] = pos[i + N * t_j];
	}
	// The positions of the particle that the current thread is interacting with are copied
	// from the global pos array into shared memory.
	// The positions are stored in separate elements for the x, y, and z 
	if(t_i < 3){
		pos_j[t_j + kABlockDimY * t_i] = pos[j + N * t_i];
	}
	// Wait for all threads to finish copying their positions into shared memory
	__syncthreads();

	// If the current thread is interacting with itself, return early
	if(i == j) return;

	// r_sqr is the square of the distance between the current particle and the particle
	// that the current thread is interacting with
	double r_sqr;
	// r3 is the cube of the distance between the current particle and the particle
  	// that the current thread is interacting with
	double r3;
	// k is a constant that is used in the calculation of the acceleration
	double k;

	// Calculate the difference between the x, y, and z coordinates of the positions
	// of the current particle and the particle that the current thread is interacting with
	double pos_diff_x = pos_j[t_j] - pos_i[t_i];
	double pos_diff_y = pos_j[t_j + kABlockDimY] - pos_i[t_i + kABlockDimX];
	double pos_diff_z = pos_j[t_j + kABlockDimY * 2] - pos_i[t_i + kABlockDimX * 2];

	// Calculate the square of the distance between the current particle and the particle
	// that the current thread is interacting with using the position differences
	r_sqr = pos_diff_x * pos_diff_x
		+ pos_diff_y * pos_diff_y
		+ pos_diff_z * pos_diff_z;
	// Calculate the cube of the distance between the particles
	r3 = sqrt(r_sqr) * r_sqr;
	// Calculate the constant k that is used in the calculation of the acceleration
	k = G * m[j] / r3;

	// Calculate the x, y, and z components of the acceleration of the current particle
	// due to the jth particle and store them in the global a array
	a[i * N + j]
		= k * pos_diff_x;
	a[i * N + j + N * N]
		= k * pos_diff_y;
	a[i * N + j + N * N * 2]
		= k * pos_diff_z;
}

// kernelVelocity is a CUDA kernel that updates the velocities and positions of the particles
// in the system based on their accelerations.
__global__ void kernelVelocity(double* m, double* a, double* v, double* pos, double* result) {
	// idx is the index of the current thread within its thread block
	int idx = threadIdx.x;
	// i is the global index of the current thread
	// This is calculated by combining the thread block indices and the thread indices
	int i = blockDim.y * blockIdx.x + threadIdx.y;

	// If the global index is out of bounds, return early
	if(i >= N){
		return;
	}

	// vx, vy, and vz are the x, y, and z components of the velocity of the current particle
	// They are initially set to 0
	double vx = 0;
	double vy = 0;
	double vz = 0;
	
	// Loop over all particles in the system
	for(int j = idx; j < N; j += 32) {
		// Add the x, y, and z components of the acceleration of the current particle
		// due to the jth particle to the corresponding velocity components
		vx += DT * a[i * N + j];
		vy += DT * a[i * N + j + N * N];
		vz += DT * a[i * N + j + N * N * 2];
	}

	// Perform a reduction on the velocity components to obtain the final values
	for(int j = 16; j > 0; j /= 2) {
		// Add the value of the velocity component of the thread j threads down in the thread block
		// to the current thread's velocity component
		vx += __shfl_down_sync(0xffffffffffffffff, vx, j);
		vy += __shfl_down_sync(0xffffffffffffffff, vy, j);
		vz += __shfl_down_sync(0xffffffffffffffff, vz, j);
	}

	// If the current thread is the first thread in the thread block, update the global
	// velocity and position arrays with the final velocity and position values for the current particle.
	if(idx == 0){
		v[i] += vx;
		v[i + N] += vy;
		v[i + N * 2] += vz;

		pos[i] += DT * v[i];
		pos[i + N] += DT * v[i + N];
		pos[i + N * 2] += DT * v[i + N * 2];
	}
}
