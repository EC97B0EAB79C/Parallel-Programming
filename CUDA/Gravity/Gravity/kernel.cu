/*
CUDA code for calculating using only CUDA thread
*/
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#include "../data_util/data_util.h"

#define N 1000
#define G 1
#define DT 1

__global__ void kernelGravity(double*, double*, double*, double*, double*);

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
	if (readDouble("../data_util/test/n1000/m.double", m) != N) {
		fprintf(stderr, "Err: Can not read m.double");
		return -1;
	}

	size = sizeof(double) * N * N * 3;
	memset(a, 0, size);

	size = sizeof(double) * N * 3;
	memset(v, 0, size);

	if (readDouble("../data_util/test/n1000/x.double", pos) != N) {
		fprintf(stderr, "Err: Can not read x.double");
		return -1;
	}
	if (readDouble("../data_util/test/n1000/y.double", pos + N) != N) {
		fprintf(stderr, "Err: Can not read y.double");
		return -1;
	}
	if (readDouble("../data_util/test/n1000/z.double", pos + N * 2) != N) {
		fprintf(stderr, "Err: Can not read z.double");
		return -1;
	}

	// device(CUDA) variable
	cudaSetDevice(0);

	double* d_m;
	size = sizeof(double) * N;
	cudaMalloc(&d_m, size);

	double* d_a;
	size = sizeof(double) * N * N * 3;
	cudaMalloc(&d_a, size);

	double* d_v;
	size = sizeof(double) * N * 3;
	cudaMalloc(&d_v, size);

	double* d_pos;
	cudaMalloc(&d_pos, size);

	double* d_result;
	cudaMalloc(&d_result, size);

	// performance metric
	cudaEvent_t start, memcpy, end;
	cudaEventCreate(&start);
	cudaEventCreate(&memcpy);
	cudaEventCreate(&end);
	cudaEventRecord(start);	// record event

	// cudaMemcpy
	size = sizeof(double) * N;
	cudaMemcpy(d_m, m, size, cudaMemcpyHostToDevice);
	size = sizeof(double) * N * N * 3;
	cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
	size = sizeof(double) * N * 3;
	cudaMemcpy(d_v, v, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_pos, pos, size, cudaMemcpyHostToDevice);
	cudaEventRecord(memcpy);	// record event

	// launch kernel
	for (int i = 0; i < 10; i++) {
		kernelGravity << <1, N >> > (d_m, d_a, d_v, d_pos, d_result);
		//cudaDeviceSynchronize();
		cudaMemcpy(d_pos, d_result, size, cudaMemcpyDeviceToDevice);
		//cudaDeviceSynchronize();
	}

	// cudaMemcpy result from device
	cudaMemcpy(pos, d_pos, size, cudaMemcpyDeviceToHost);
	cudaEventRecord(end);	// record event
	cudaDeviceSynchronize();

	// print performace metrics
	float milisecondsStoM = 0;
	cudaEventElapsedTime(&milisecondsStoM, start, memcpy);
	fprintf(stdout, "Time from start to memcpy:\n\t%f\n", milisecondsStoM);
	float milisecondsStoE = 0;
	cudaEventElapsedTime(&milisecondsStoE, start, end);
	fprintf(stdout, "Time from start to end:\n\t%f\n", milisecondsStoE);

	// write results
	char writeFileX[] = "./cuda_x.double";
	writeDouble(writeFileX, pos, N);
	char writeFileY[] = "./cuda_y.double";
	writeDouble(writeFileY, pos + N, N);
	char writeFileZ[] = "./cuda_z.double";
	writeDouble(writeFileZ, pos + N * 2, N);

	// free memory
	cudaFree(d_m);
	cudaFree(d_a);
	cudaFree(d_v);
	cudaFree(d_pos);
	cudaFree(d_result);

	free(m);
	free(a);
	free(v);
	free(pos);

	cudaDeviceReset();

	return 0;
}

__global__ void kernelGravity(double* m, double* a, double* v, double* pos, double* result) {
	int i = threadIdx.x;
	double r_sqr;
	double r;
	for (int j = 0; j < N; j++) {
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

	for (int j = 0; j < N; j++) {
		if (i != j) {
			v[i] += DT * a[i * N + j];
			v[i + N] += DT * a[i * N + j + N * N];
			v[i + N * 2] += DT * a[i * N + j + N * N * 2];
		}
	}

	result[i] = pos[i] + DT * v[i];
	result[i + N] = pos[i + N] + DT * v[i + N];
	result[i + N * 2] = pos[i + N * 2] + DT * v[i + N * 2];

}