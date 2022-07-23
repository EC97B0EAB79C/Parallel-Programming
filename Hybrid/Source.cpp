#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include<random>
#include<sys/stat.h>
#include<sys/time.h>

#include<omp.h>
#include<mpi.h>

#include "data_util_bin.c"

#define N 10000
#define G 1.0L
#define DT 1.0L
#define ITER 10

typedef struct double3_
{
	double x, y, z;
} double3;

double 	get_time();
void export_double3(double3* data, int size);

int main() {
	int ctr;
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
	double3* pos_next;
	pos_next = (double3*)malloc(sizeof(double3) * N);
	if (pos_next == NULL) {
		fprintf(stderr, "ERR: malloc \"pos_next\"\n");
		exit(1);
	}
	memset(m, 0, sizeof(double) * N);
	memset(a, 0, sizeof(double3) * N * N);
	memset(v, 0, sizeof(double3) * N);
	memset(pos, 0, sizeof(double3) * N);
	memset(pos_next, 0, sizeof(double3) * N);
	double start, prev, end;
	
	//initialize MPI
	int mpi_size;
	int mpi_rank;
	MPI_Init(NULL,NULL);
	MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
	MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
	//create type MPI_DOUBLE3
	MPI_Datatype MPI_DOUBLE3;
	int array_of_blocklengths[3]={1,1,1};
	MPI_Aint array_of_displacements[3];	
	array_of_displacements[0]=offsetof(double3, x);
	array_of_displacements[1]=offsetof(double3, y);
	array_of_displacements[2]=offsetof(double3, z);
	MPI_Datatype array_of_types[3]={MPI_DOUBLE,MPI_DOUBLE,MPI_DOUBLE};
	MPI_Type_create_struct(3,array_of_blocklengths,array_of_displacements,array_of_types,
							&MPI_DOUBLE3);
	MPI_Type_commit(&MPI_DOUBLE3);//fprintf(stdout,"MPI init passed");
	
	
	//MPI variables
	//array of calculating size per process
	int* calc_counts=(int*)malloc(sizeof(int)*mpi_size);
	if (calc_counts == NULL) {
		fprintf(stderr, "ERR: malloc \"calc_counts\"\n");
		exit(1);
	}
	//array of calculation displacement per process
	int* displs=(int*)malloc(sizeof(int)*mpi_size);
	if (displs == NULL) {
		fprintf(stderr, "ERR: malloc \"displs\"\n");
		exit(1);
	}
	//process with rank less than N%size calculates N/size+1 
	//process with rank more than N%size calculates N/size 
	calc_counts[0]=N/mpi_size+(0<N%mpi_size);
	for(ctr=0;ctr<mpi_size;ctr++){
		calc_counts[ctr]=N/mpi_size+(ctr<N%mpi_size);
		displs[ctr]=displs[ctr-1]+calc_counts[ctr-1];
	}//fprintf(stdout,"mpi variable passed");
	
	
	//create(load) initial data
	if(mpi_rank==0){
		double* temp=(double*)malloc(sizeof(double)*N);
		read_data("/path/to/file/m.double", m, N);
		read_data("/path/to/file/x.double", temp, N);
		for(ctr=0;ctr<N;ctr++){
			pos[ctr].x=temp[ctr];
			pos_next[ctr].x=pos[ctr].x;
		}
		read_data("/path/to/file/y.double", temp, N);
		for(ctr=0;ctr<N;ctr++){
			pos[ctr].y=temp[ctr];
			pos_next[ctr].y=pos[ctr].y;
		}
		read_data("/path/to/file/z.double", temp, N);
		for(ctr=0;ctr<N;ctr++){
			pos[ctr].z=temp[ctr];
			pos_next[ctr].z=pos[ctr].z;
		}
		read_data("/path/to/file/vx.double", temp, N);
		for(ctr=0;ctr<N;ctr++){
			v[ctr].x=temp[ctr];
		}
		read_data("/path/to/file/vy.double", temp, N);
		for(ctr=0;ctr<N;ctr++){
			v[ctr].y=temp[ctr];
		}
		read_data("/path/to/file/vz.double", temp, N);
		for(ctr=0;ctr<N;ctr++){
			v[ctr].z=temp[ctr];
		}
		
		free(temp);
	}
	
	
	//broadcast initial data
	MPI_Bcast(m,N,MPI_DOUBLE,0,MPI_COMM_WORLD);
	MPI_Bcast(v,N,MPI_DOUBLE3,0,MPI_COMM_WORLD);
	MPI_Bcast(pos,N,MPI_DOUBLE3,0,MPI_COMM_WORLD);
	MPI_Bcast(pos_next,N,MPI_DOUBLE3,0,MPI_COMM_WORLD);
		
	//ctr and temporary variable
	int i, j, k, iter = 0;
	double r_sqr;
	double r_sqrt;
	double temp_reduct;
	start = get_time();
	prev = start;
	while (iter < ITER) {

#pragma omp parallel private(i,j,k,r_sqr,r_sqrt)
		{	
			#pragma omp for
			for (i = displs[mpi_rank]; i < displs[mpi_rank]+calc_counts[mpi_rank]; i++) {
				//calculate acceleration 
				for (j = 0; j < N; j++) {
					r_sqr = (pos[i].x - pos[j].x) * (pos[i].x - pos[j].x) + (pos[i].y - pos[j].y) * (pos[i].y - pos[j].y) + (pos[i].z - pos[j].z) * (pos[i].z - pos[j].z);
					r_sqrt= sqrt(r_sqr);
					a[i * N + j].x = G * ((m[j]) / (r_sqr)) * ((pos[j].x - pos[i].x) / (r_sqrt));
					a[i * N + j].y = G * ((m[j]) / (r_sqr)) * ((pos[j].y - pos[i].y) / (r_sqrt));
					a[i * N + j].z = G * ((m[j]) / (r_sqr)) * ((pos[j].z - pos[i].z) / (r_sqrt));
				}
				//calculate speed
				for (j = 0; j < N; j++) {
					if (i != j) {
						v[i].x += DT * a[i * N + j].x;
						v[i].y += DT * a[i * N + j].y;
						v[i].z += DT * a[i * N + j].z;
					}
				}
				//calculate position
				pos_next[i].x += DT * v[i].x;
				pos_next[i].y += DT * v[i].y;
				pos_next[i].z += DT * v[i].z;	
			}
		}

		//Allgather calculated positions
		MPI_Allgatherv(&pos_next[displs[mpi_rank]], calc_counts[mpi_rank], MPI_DOUBLE3,
						pos, calc_counts, displs, MPI_DOUBLE3, MPI_COMM_WORLD);
		end = get_time();

		prev=end;
		iter++;
	}
	//print runtime
	if(mpi_rank==0){
		fprintf(stdout, "%f\n", end - start);
	}

	
	//free
	free(m);
	free(a);
	free(v);
	free(pos);
	free(pos_next);
	free(calc_counts);
	free(displs);
	
}

double get_time() {
	struct timeval tm;
	double t;

	gettimeofday(&tm, NULL);
	t = (double)(tm.tv_sec) + ((double)(tm.tv_usec)) / 1.0e6;
	return t;
}

void export_double3(double3* data, int size){
	double* temp=(double*)malloc(sizeof(double)*size);
	if(temp==NULL){
		fprintf(stderr, "ERR: malloc \"temp\"\n");
		exit(1);
	}
	int i;
	for(i=0;i<N;i++)
		temp[i]=data[i].x;
	write_data(temp,"/work/EDU3/choi/homework/p04/mpi/result/calcx.dat",size);
	for(i=0;i<N;i++)
		temp[i]=data[i].y;
	write_data(temp,"/work/EDU3/choi/homework/p04/mpi/result/calcy.dat",size);
	for(i=0;i<N;i++)
		temp[i]=data[i].z;
	write_data(temp,"/work/EDU3/choi/homework/p04/mpi/result/calcz.dat",size);
	free(temp);
	fprintf(stdout, "file saved\n");
	return;
}

