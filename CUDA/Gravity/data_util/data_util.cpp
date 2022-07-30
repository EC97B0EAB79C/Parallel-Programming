#include<stdio.h>
#include<stdlib.h>

#include "./data_util.h"

// Read from binary double file
long readDouble(char* fileName, double* data) {

	// File variable and open
	FILE* file;
	if ((file = fopen(fileName, "rb")) == NULL) {
		fprintf(stderr, "Err: Failed to open File(%s)\n", fileName);
		return -1;
	}

	// Get file size
	size_t fileSize;
	fseek(file, 0, SEEK_END);
	fileSize = ftell(file);
	rewind(file);

	// Read data from file
	size_t readSize = 0;
	while (readSize < fileSize) { // 
		readSize += fread(
			data + readSize,
			1,
			fileSize - readSize,
			file);
		if (ferror(file) || feof(file)) {
			fprintf(stderr, "Err: Failed to read File(%s)\n", fileName);
			fclose(file);
			return -1;
		}
	}

	fclose(file);
	return fileSize / sizeof(double);
}

// Write to binary double file
long writeDouble(char* fileName, double* data, long size) {

	// File variable and open
	FILE* file;
	if ((file = fopen(fileName, "wb")) == NULL) {
		fprintf(stderr, "Err: Failed to open File(%s)\n", fileName);
		return -1;
	}

	// Target file size
	size_t fileSize = size * sizeof(double);

	// Write data to file
	size_t writeSize = 0;
	while (writeSize < fileSize) {
		writeSize += fwrite(data + writeSize, 1, fileSize - writeSize, file);
		if (ferror(file)) {
			fprintf(stderr, "Err: Failed to write File(%s)\n", fileName);
			return -1;
		}
	}

	return size;
}