#pragma once	

// Read from binary double file 
long readDouble(char* fileName, double* data);

// Write to binary double file
long writeDouble(char* fileName, double* data, long size);
