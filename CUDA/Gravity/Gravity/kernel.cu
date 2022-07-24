/*
CUDA code for calculating using only CUDA thread
*/
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include<math.h>

#define N 1000
#define G 10
#define DT .1

__global__ void kernelGravity(double*, double*, double*, double*);

int main()
{
	cudaSetDevice(0);

	size_t size;

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

	//TODO	cudaMemcpy

	kernelGravity << <1, N >> > (d_m, d_a, d_v, d_pos);

	cudaDeviceReset();

	return 0;
}

__global__ void kernelGravity(double* m, double* a, double* v, double* pos) {
	int i = threadIdx.x;
	double r_sqr;
	for (int j = 0; j < N; j++) {
		r_sqr = (pos[i] - pos[j]) * (pos[i] - pos[j])
			+ (pos[i + N] - pos[j + N]) * (pos[i + N] - pos[j + N])
			+ (pos[i + N * 2] - pos[j + N * 2]) * (pos[i + N * 2] - pos[j + N * 2]);

		a[i * N + j] = G * (m[j]) / (r_sqr)
			* (pos[i] - pos[j]) / (sqrt(r_sqr));
		a[i * N + j + N * N] = G * (m[j]) / (r_sqr)
			* (pos[i + N] - pos[j + N]) / (sqrt(r_sqr));
		a[i * N + j + N * N * 2] = G * (m[j]) / (r_sqr)
			* (pos[i + N * 2] - pos[j + N * 2]) / (sqrt(r_sqr));
	}

	for (int j = 0; j < N; j++) {
		if (i != j) {
			v[i] += DT * a[i * N + j];
			v[i + N] += DT * a[i * N + j + N * N];
			v[i + N * 2] += DT * a[i * N + j + N * N * 2];
		}
	}

	__syncthreads();

	pos[i] += DT * v[i];
	pos[i + N] += DT * v[i + N];
	pos[i + N * 2] += DT * v[i + N * 2];

}