/*
Source file to test "data_compare"
*/
#include<stdio.h>
#include<stdlib.h>

#include "data_compare.h"

int main(int argc, char* argv[]) {

	if (argc < 3) {
		fprintf(stderr, "Usage:\n\t%s [file1] [file2]\n", argv[0]);
		return -1;
	}

	long result;
	if ((result = compareDouble(argv[1], argv[2])) < 0) {
		fprintf(stderr, "Err: Failed to compare two file (%s) and (%s)\n", argv[1], argv[2]);
		return -1;
	}

	if (result == 0) {
		fprintf(stdout, "Every data's error is in less than %1.20lf.\n", 1e-10);
	}
	else {
		fprintf(stdout, "There are %1ld diffrent datas\n", result);
	}

}
