/*
CUDA code for calculating using CUDA thread blocks and CUDA threads
	Divide kernel in steps of Acceleration, Velocity, Position
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

__global__ void kernelAcceleration(double*, double*, double*, double*, double*);
__global__ void kernelVelocity(double*, double*, double*, double*, double*);
__global__ void kernelPosition(double*, double*, double*, double*, double*);


cudaError runKernel(double* m, double* a, double* v, double* pos) {
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
	dim3 blockDim;
	dim3 gridDim;
	for (int i = 0; i < 10; i++) {
		blockDim = dim3(16, 16);
		gridDim = dim3((N + (blockDim.x - 1)) / blockDim.x, (N + (blockDim.y - 1)) / blockDim.y);
		kernelAcceleration << <gridDim, blockDim >> > (d_m, d_a, d_v, d_pos, d_result);
		/*
		cudaStatus = cudaDeviceSynchronize();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "Err: %dth iter Kernel\n", i);
			goto Error;
		}
		*/

		blockDim = dim3(256);
		gridDim = dim3((N + (blockDim.x - 1)) / blockDim.x);
		kernelVelocity << <gridDim, blockDim >> > (d_m, d_a, d_v, d_pos, d_result);
		/*
		cudaStatus = cudaDeviceSynchronize();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "Err: %dth iter Kernel\n", i);
			goto Error;
		}
		*/

		kernelPosition << <gridDim, blockDim >> > (d_m, d_a, d_v, d_pos, d_result);
		/*
		cudaStatus = cudaDeviceSynchronize();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "Err: %dth iter Kernel\n", i);
			goto Error;
		}
		*/

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
	float milisecondsStoM = 0;
	cudaEventElapsedTime(&milisecondsStoM, start, memcpy);
	fprintf(stdout, "Time from start to memcpy:\n\t%f\n", milisecondsStoM);
	float milisecondsStoE = 0;
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


int main()
{
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
		fprintf(stderr, "Err: Malloc Failed");
		return -1;
	}


	// set variable
	if (readDouble("../data_util/test/n10000/m.double", m) != N) {
		fprintf(stderr, "Err: Can not read m.double");
		return -1;
	}

	size = sizeof(double) * N * N * 3;
	memset(a, 0, size);

	size = sizeof(double) * N * 3;
	memset(v, 0, size);

	if (readDouble("../data_util/test/n10000/x.double", pos) != N) {
		fprintf(stderr, "Err: Can not read x.double");
		return -1;
	}
	if (readDouble("../data_util/test/n10000/y.double", pos + N) != N) {
		fprintf(stderr, "Err: Can not read y.double");
		return -1;
	}
	if (readDouble("../data_util/test/n10000/z.double", pos + N * 2) != N) {
		fprintf(stderr, "Err: Can not read z.double");
		return -1;
	}


	// run kernel
	cudaError_t cudaStatus;
	if ((cudaStatus = runKernel(m, a, v, pos)) != cudaSuccess) {
		fprintf(stderr, "Err: Kernal returned error code %d\n\t%s\n",
			cudaStatus,
			cudaGetErrorString(cudaStatus));
	}

	printf("%lf", pos[9999]);
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
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	int j = blockDim.y * blockIdx.y + threadIdx.y;
	if (i >= N || j >= N) {
		return;
	}

	double r_sqr;
	double r;

	r_sqr
		= (pos[i] - pos[j]) * (pos[i] - pos[j])
		+ (pos[i + N] - pos[j + N]) * (pos[i + N] - pos[j + N])
		+ (pos[i + N * 2] - pos[j + N * 2]) * (pos[i + N * 2] - pos[j + N * 2]);
	r = sqrt(r_sqr);
	a[i * N + j]
		= G * (m[j]) / (r_sqr) * (pos[j] - pos[i]) / r;
	a[i * N + j + N * N]
		= G * (m[j]) / (r_sqr) * (pos[j + N] - pos[i + N]) / r;
	a[i * N + j + N * N * 2]
		= G * (m[j]) / (r_sqr) * (pos[j + N * 2] - pos[i + N * 2]) / r;
}

__global__ void kernelVelocity(double* m, double* a, double* v, double* pos, double* result) {
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i >= N) {
		return;
	}

	for (int j = 0; j < N; j++) {
		if (i != j) {
			v[i] += DT * a[i * N + j];
			v[i + N] += DT * a[i * N + j + N * N];
			v[i + N * 2] += DT * a[i * N + j + N * N * 2];
		}
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