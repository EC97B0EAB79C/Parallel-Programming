#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include<random>
#include<sys/time.h>

#include<omp.h>

#define N 1000
#define G 9.8
#define DT .1
#define ITER 100

double 	get_time();

typedef struct double3_
{
	double x, y, z;
} double3;

int main() {
	//malloc variables
	double* m;
	m = (double*)malloc(sizeof(double) * N);
	if (m == NULL) {
		fprintf(stderr, "ERR: malloc \"m\"\n");
		exit(1);
	}
	double3* a;
	a = (double3*)malloc(sizeof(double3) * N * N);
	if (a == NULL) {
		fprintf(stderr, "ERR: malloc \"a\"\n");
		exit(1);
	}
	double3* v;
	v = (double3*)malloc(sizeof(double3) * N);
	if (v == NULL) {
		fprintf(stderr, "ERR: malloc \"v\"\n");
		exit(1);
	}
	double3* pos;
	pos = (double3*)malloc(sizeof(double3) * N);
	if (pos == NULL) {
		fprintf(stderr, "ERR: malloc \"pos\"\n");
		exit(1);
	}
	float r_sqr;
	double start, prev, end;

	//initialize variables
	srand(0);
	for (int i = 0; i < N; i++) {
		m[i] = 1. + (double)rand() ;
		pos[i].x = (double)rand() / RAND_MAX * 200 - 100;
		pos[i].y = (double)rand() / RAND_MAX * 200 - 100;
		pos[i].z = (double)rand() / RAND_MAX * 200 - 100;
	}
	memset(a, 0, sizeof(double) * N * N);
	memset(v, 0, sizeof(double) * N);

	//ctr
	int i, j, k, iter = 0;

	start = get_time();
	prev = start;
	while (iter++ < ITER) {

#pragma omp parallel private(i,j,k,r_sqr)
		{
#pragma omp for
			for (i = 0; i < N; i++) {
				//calculate acceleration 
				for (j = 0; j < N; j++) {
					r_sqr = (pos[i].x - pos[j].x) * (pos[i].x - pos[j].x) + (pos[i].y - pos[j].y) * (pos[i].y - pos[j].y) + (pos[i].z - pos[j].z) * (pos[i].z - pos[j].z);
					a[i * N + j].x = G * (m[j]) / (r_sqr) * (pos[i].x - pos[j].x) / (sqrt(r_sqr));
					a[i * N + j].y = G * (m[j]) / (r_sqr) * (pos[i].y - pos[j].y) / (sqrt(r_sqr));
					a[i * N + j].z = G * (m[j]) / (r_sqr) * (pos[i].z - pos[j].z) / (sqrt(r_sqr));
				}
			}

#pragma omp for
			for (i = 0; i < N; i++) {
				//calculate speed
				for (j = 0; j < N; j++) {
					if (i != j) {
						v[i].x += DT * a[i * N + j].x;
						v[i].y += DT * a[i * N + j].y;
						v[i].z += DT * a[i * N + j].z;
					}
				}
				//calculate position
				pos[i].x += DT * v[i].x;
				pos[i].y += DT * v[i].y;
				pos[i].z += DT * v[i].z;
			}
		}
		end = get_time();
//		fprintf(stdout, "iter\t\t%d\ttime elapsed: %f\n", iter, end - prev);
		prev=end;
	}
	fprintf(stdout, "final\ttime elapsed: %f\n", end - start);
}

double get_time() {
	struct timeval tm;
	double t;

	gettimeofday(&tm, NULL);
	t = (double)(tm.tv_sec) + ((double)(tm.tv_usec)) / 1.0e6;
	return t;
}