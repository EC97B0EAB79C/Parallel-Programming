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

__global__ void kernelAcceleration(double*, double*, double*, double*, double*);
__global__ void kernelVelocity(double*, double*, double*, double*, double*);
__global__ void kernelPosition(double*, double*, double*, double*, double*);


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
		/*
		*/
		cudaStatus = cudaDeviceSynchronize();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "Err: %dth iter Kernel\n", i);
			goto Error;
		}

		blockDim = dim3(256);
		gridDim = dim3((N + (blockDim.x - 1)) / blockDim.x);
		kernelVelocity << <gridDim, blockDim >> > (d_m, d_a, d_v, d_pos, d_result);
		/*
		*/
		cudaStatus = cudaDeviceSynchronize();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "Err: %dth iter Kernel\n", i);
			goto Error;
		}

		kernelPosition << <gridDim, blockDim >> > (d_m, d_a, d_v, d_pos, d_result);
		/*
		*/
		cudaStatus = cudaDeviceSynchronize();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "Err: %dth iter Kernel\n", i);
			goto Error;
		}

		cudaStatus = cudaMemcpy(d_pos, d_result, size, cudaMemcpyDeviceToDevice);
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

__global__ void kernelAcceleration(double* m, double* a, double* v, double* pos, double* result) {
	int t_i = threadIdx.x;
	int t_j = threadIdx.y;
	int i = blockDim.x * blockIdx.x + t_i;
	int j = blockDim.y * blockIdx.y + t_j;
	if (i >= N || j >= N) return;

	__shared__ double pos_i[kABlockDimX * 3];
	__shared__ double pos_j[kABlockDimY * 3];
	
	if(t_j < 3){
		pos_i[t_i + kABlockDimX * t_j] = pos[i + N * t_j];
	}
	if(t_i < 3){
		pos_j[t_j + kABlockDimY * t_i] = pos[j + N * t_i];
	}
	__syncthreads();

	if(i == j) return;

	double r_sqr;
	double r3;
	double k;
//TODO
	double pos_diff_x = pos_j[t_j] - pos_i[t_i];
	double pos_diff_y = pos_j[t_j + kABlockDimY] - pos_i[t_i + kABlockDimX];
	double pos_diff_z = pos_j[t_j + kABlockDimY * 2] - pos_i[t_i + kABlockDimX * 2];

	r_sqr = pos_diff_x * pos_diff_x
		+ pos_diff_y * pos_diff_y
		+ pos_diff_z * pos_diff_z; 
	r3 = sqrt(r_sqr) * r_sqr;
	k = G * m[j] / r3;

	a[i * N + j]
		= k * pos_diff_x;
	a[i * N + j + N * N]
		= k * pos_diff_y;
	a[i * N + j + N * N * 2]
		= k * pos_diff_z;
/*
	if((pos_i[t_i] - pos_j[t_j]) != (pos[i] - pos[j])) printf("err");
	r_sqr	= (pos_i[t_i] - pos_j[t_j]) * (pos[i] - pos[j])
		+ (pos[i + N] - pos[j + N]) * (pos[i + N] - pos[j + N])
		+ (pos[i + N * 2] - pos[j + N * 2]) * (pos[i + N * 2] - pos[j + N * 2]);
	r3 = sqrt(r_sqr) * r_sqr;
	k = G * m[j] / r3;

	a[i * N + j]
		= k * (pos[j] - pos[i]);
	a[i * N + j + N * N]
		= k * (pos[j + N] - pos[i + N]);
	a[i * N + j + N * N * 2]
		= k * (pos[j + N * 2] - pos[i + N * 2]);
*/
/*
*/
}

__global__ void kernelVelocity(double* m, double* a, double* v, double* pos, double* result) {
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i >= N) {
		return;
	}

/*	cuda data initilization no need?	
	a[i*N+i] = 0;
	a[i*N+i+N*N] = 0;
	a[i*N+i+N*N*2] = 0;
*/

	for (int j = 0; j < N; j++) {
		v[i] += DT * a[i * N + j];
		v[i + N] += DT * a[i * N + j + N * N];
		v[i + N * 2] += DT * a[i * N + j + N * N * 2];
	}
}

__global__ void kernelPosition(double* m, double* a, double* v, double* pos, double* result) {
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i >= N) {
		return;
	}

	result[i] = pos[i] + DT * v[i];
	result[i + N] = pos[i + N] + DT * v[i + N];
	result[i + N * 2] = pos[i + N * 2] + DT * v[i + N * 2];
}