//
// cuda_kernel_tools.cu
// GPUMPM_HKM
//
// created by Kemeng Huang on 2020/05/23
// Copyright (c) 2020 Kemeng Huang. All rights reserved.
//

#include"cuda_kernal_tools.cuh"
#include"cuda_tools.h"

#define threads_size 256
int* recursive_sum;

__global__
void In_Prefix_Sum_Kernal(int* cell_offset, int* out_offset, int* recursive_sum, int cellNum)
{
	int ds = blockDim.x << 1;
	int idof = __mul24(ds, blockIdx.x);
	int x_idp = threadIdx.x + idof;
	extern __shared__ int data[];
	if (x_idp >= cellNum) return;
	data[threadIdx.x] = cell_offset[x_idp];
	int nextDb = x_idp + blockDim.x;
	if (nextDb < cellNum) {
		data[threadIdx.x + blockDim.x] = cell_offset[nextDb];
	}
	__syncthreads();
	int iSize = 1;
	int b = 0;
	int gpSize, gpId, x_id, P;
	while (iSize < ds)
	{
		gpSize = iSize;
		iSize <<= 1;
		gpId = (threadIdx.x >> b) + 1;
		x_id = (gpId << b) + threadIdx.x;
		b++;
		P = ((gpId - 1) << b) + (gpSize - 1);
		if (x_id + idof < cellNum) {
			data[x_id] += data[P];
		}
		__syncthreads();
	}
	out_offset[x_idp] = data[threadIdx.x];
	if (nextDb < cellNum) {
		out_offset[nextDb] = data[threadIdx.x + blockDim.x];
	}
	if (threadIdx.x == (blockDim.x - 1)) {
		recursive_sum[blockIdx.x] = data[ds - 1];
	}
}

__global__
void Ex_Prefix_Sum_Kernal(int* cell_offset, int* out_offset, int* recursive_sum, int cellNum)
{
	int ds = blockDim.x << 1;
	int idof = __mul24(ds, blockIdx.x);
	int x_idp = threadIdx.x + idof;
	extern __shared__ int data[];
	if (x_idp >= cellNum) return;
	data[threadIdx.x] = cell_offset[x_idp];
	int nextDb = x_idp + blockDim.x;
	if (nextDb < cellNum) {
		data[threadIdx.x + blockDim.x] = cell_offset[nextDb];
	}

	__syncthreads();
	int iSize = 1;
	int b = 0;
	int gpSize, gpId, x_id, P;
	while (iSize < ds)
	{
		gpSize = iSize;
		iSize <<= 1;

		gpId = (threadIdx.x >> b) + 1;
		x_id = (gpId << b) + threadIdx.x;

		b++;

		P = ((gpId - 1) << b) + (gpSize - 1);

		if (x_id + idof < cellNum) {

			data[x_id] += data[P];

		}
		__syncthreads();
	}
	out_offset[x_idp+1] = data[threadIdx.x];
	if (nextDb < cellNum) {
		out_offset[nextDb+1] = data[threadIdx.x + blockDim.x];
	}
	if (threadIdx.x == (blockDim.x - 1)) {
		recursive_sum[blockIdx.x] = data[ds - 1];
	}
}

__global__
void In_Prefix_Sum_Kernal_Recursive(int* cell_offset, int* recursive_sum, int offset, int recursive_off, int cellNum)
{
	int ds = blockDim.x << 1;
	int idof = __mul24(ds, blockIdx.x);
	int x_idp = threadIdx.x + idof;
	extern __shared__ int data[];
	if (x_idp >= cellNum) return;
	data[threadIdx.x] = cell_offset[x_idp + offset];
	int nextDb = x_idp + blockDim.x;
	if (nextDb < cellNum) {
		data[threadIdx.x + blockDim.x] = cell_offset[nextDb + offset];
	}
	__syncthreads();
	int iSize = 1;
	int b = 0;
	int gpSize, gpId, x_id, P;
	while (iSize < ds)
	{
		gpSize = iSize;
		iSize <<= 1;
		gpId = (threadIdx.x >> b) + 1;
		x_id = (gpId << b) + threadIdx.x;
		b++;
		P = ((gpId - 1) << b) + (gpSize - 1);
		if (x_id + idof < cellNum) {
			data[x_id] += data[P];
		}
		__syncthreads();
	}

	cell_offset[x_idp + offset] = data[threadIdx.x];
	if (nextDb < cellNum) {
		cell_offset[nextDb + offset] = data[threadIdx.x + blockDim.x];
	}
	if (threadIdx.x == (blockDim.x - 1)) {
		recursive_sum[blockIdx.x + recursive_off] = data[ds - 1];
	}
}

__global__
void addOffset(int* cell_offset, int* recursive_sum, int recursive_off, int cellNum)
{
	int x_idp = __mul24(blockDim.x, (blockIdx.x+1)) + threadIdx.x;
	if (x_idp >= cellNum) return;
	cell_offset[x_idp] += recursive_sum[recursive_off + blockIdx.x];
}

void Allocate_Prefix_Sum_RecursiveMem_Int(int num) {
	int dataSize = threads_size << 1;
	int blockNum_temp = (num + (dataSize - 1)) / (dataSize);
	int recursive_size = 1;
	while (blockNum_temp > 1) {
		recursive_size += blockNum_temp;
		blockNum_temp = (blockNum_temp + dataSize - 1) / dataSize;
	}
	CUDA_SAFE_CALL(cudaMalloc((void**)&recursive_sum, recursive_size * sizeof(int)));
}

void Free_Prefix_Sum_RecursiveMem_Int() {
	CUDA_SAFE_CALL(cudaFree(recursive_sum));
}

void In_Prefix_Sum_Recursive_Int(int offsetI, const int& blockNum, int* inputArray, const int& Number, const  int& offset)
{
	int dataSize = threads_size << 1;
	unsigned int memSize = sizeof(int) * dataSize;
	In_Prefix_Sum_Kernal_Recursive << <blockNum, threads_size, memSize >> > (inputArray, recursive_sum, offsetI, offset, Number);
	if (blockNum == 1) {
		return;
	}
	int newBlockNum = (blockNum + (dataSize - 1)) / dataSize;
	int newOffset = offset + blockNum;
	In_Prefix_Sum_Recursive_Int(offset, newBlockNum, recursive_sum, blockNum, newOffset);
	addOffset << <blockNum - 1, dataSize >> > (inputArray, recursive_sum, offset, Number);
}

void Ex_Prefix_Sum_Int(int* inputArray, int* outputArray, const int& Number)
{
	int dataSize = threads_size << 1;
	int blockNum = (Number + (dataSize - 1)) / dataSize;
	unsigned int memSize = sizeof(int) * dataSize;
	Ex_Prefix_Sum_Kernal << <blockNum, threads_size, memSize >> > (inputArray, outputArray, recursive_sum, Number);
	if (blockNum == 1) {
		return;
	}
	int newBlockNum = (blockNum + (dataSize - 1)) / dataSize;
	In_Prefix_Sum_Recursive_Int(0, newBlockNum, recursive_sum, blockNum, blockNum);
	addOffset << <blockNum - 1, dataSize >> > (outputArray + 1, recursive_sum, 0, Number);
}

void In_Prefix_Sum_Int(int* inputArray, int* outputArray, const int& Number)
{
	int dataSize = threads_size << 1;
	int blockNum = (Number + (dataSize - 1)) / dataSize;
	unsigned int memSize = sizeof(int) * dataSize;
	In_Prefix_Sum_Kernal << <blockNum, threads_size, memSize >> > (inputArray, outputArray, recursive_sum, Number);
	if (blockNum == 1) {
		return;
	}
	int newBlockNum = (blockNum + (dataSize - 1)) / dataSize;
	In_Prefix_Sum_Recursive_Int(0, newBlockNum, recursive_sum, blockNum, blockNum);
	addOffset << <blockNum - 1, dataSize >> > (outputArray, recursive_sum, 0, Number);
}

