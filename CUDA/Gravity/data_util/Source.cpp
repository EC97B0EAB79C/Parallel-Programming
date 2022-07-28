/*
Source file to test "data_util"
*/
#include<stdio.h>
#include<stdlib.h>

#include "data_util.h"

int main(int argc, char* argv[]) {

	int n = 1000;

	if (argc<2) {
		fprintf(stderr, "Usage:\n\t%s [filename]\n",argv[0]);
		return -1;
	}

	double* temp = (double*)malloc(sizeof(double) * n);
	if (temp == NULL) {
		fprintf(stderr, "Err: Failed to malloc\n");
		return -1;
	}

	if (readDouble(argv[1], temp)>0) {
		for (int i = 0; i < n; i++) {
			fprintf(stdout, "%lf\n", temp[i]);
		}
	}
	else {
		fprintf(stderr, "Err: Failed to read file\n");
		return -1;
	}

	char writeFile[] = "./test.double";
	if (writeDouble(
		writeFile,
		temp,
		n)) {
		fprintf(stdout, "Write Success");
	}
	else {
		fprintf(stderr, "Err: Failed to write file\n");
		return -1;
	}


}