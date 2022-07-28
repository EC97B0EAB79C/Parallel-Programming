#include<stdio.h>
#include<stdlib.h>
#include<cmath>

#include "data_compare.h"

double compareDouble(char* fileName1, char* fileName2)
{
	// File1
	FILE* file1;
	// open file
	if ((file1 = fopen(fileName1, "rb")) == NULL) {
		fprintf(stderr, "Err: Failed to open File(%s)\n", fileName1);
		return -1;
	}
	// file size
	size_t fileSize1;
	fseek(file1, 0, SEEK_END);
	fileSize1 = ftell(file1);
	rewind(file1);

	// File2
	FILE* file2;
	// open file
	if ((file2 = fopen(fileName2, "rb")) == NULL) {
		fprintf(stderr, "Err: Failed to open File(%s)\n", fileName2);
		return -1;
	}
	// file size
	size_t fileSize2;
	fseek(file2, 0, SEEK_END);
	fileSize2 = ftell(file2);
	rewind(file2);

	// Check if two file have same amount of data
	if (fileSize1 != fileSize2) {
		fprintf(
			stderr,
			"Err: Two file (%s) and (%s) have different sizes\n",
			fileName1,
			fileName2);
		return -1;
	}

	// Read data can compare
	long numElements = fileSize1 / sizeof(double);
	double difference;
	double totalError = 0;
	double buff1, buff2;
	for (long i = 0; i < numElements; i++) {
		// read a double from File1
		fread(&buff1, sizeof(double), 1, file1);
		if (ferror(file1) || feof(file1)) {
			fprintf(stderr, "Err: Failed to read File(%s)\n", fileName1);
			return -1;
		}

		// read a double from File2
		fread(&buff2, sizeof(double), 1, file2);
		if (ferror(file2) || feof(file2)) {
			fprintf(stderr, "Err: Failed to read File(%s)\n", fileName2);
			return -1;
		}
		
		// compare two double
		if ((difference = abs(buff1 - buff2)) > 1e-10) {
			fprintf(stdout, "%ld has different result\n\tdifference: %lf\n", i, difference);
		}

		totalError += abs(buff1 - buff2);
	}

	return totalError;
}
