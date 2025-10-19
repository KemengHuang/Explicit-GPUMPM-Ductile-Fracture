#include"Transformer.cuh"
#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include"cuda_tools.h"
#include"cuda_kernal_tools.cuh"
__device__ unsigned long long Packed_Add(const unsigned long long* masks, const unsigned long long i, const unsigned long long j) {
    unsigned long long x_result = ((i | ~masks[0]) + (j & masks[0])) & masks[0];
    unsigned long long y_result = ((i | ~masks[1]) + (j & masks[1])) & masks[1];
    unsigned long long z_result = ((i | ~masks[2]) + (j & masks[2])) & masks[2];
    unsigned long long w_result = ((i | masks[0] | masks[1] | masks[2]) + (j & ~(masks[0] | masks[1] | masks[2]))) & ~(masks[0] | masks[1] | masks[2]);
    unsigned long long result = x_result | y_result | z_result | w_result;
    return result;
}

__global__ void markCellBoundary(const int numParticle, const unsigned long long* _offsets, int* _marks) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numParticle) return;
    if (!idx || _offsets[idx] != _offsets[idx - 1])
        _marks[idx] = 1;
    else
        _marks[idx] = 0;
    if (idx == 0)
        _marks[numParticle] = 1;
}

__global__ void markBlockOffset(const int numParticle, int* _blockMap, int* _particleOffsets) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numParticle) return;
    int mapid = _blockMap[idx];
    if (!idx || mapid != _blockMap[idx - 1]) {
        _particleOffsets[mapid - 1] = idx;
    }
    /// mark sentinel
    if (idx == numParticle - 1)
        _particleOffsets[mapid] = numParticle;
}

__global__ void markPageBoundary(const int numParticle, const unsigned long long* _offsets, int* _marks) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numParticle) return;
    if (!idx || (_offsets[idx] >> 12) != (_offsets[idx - 1] >> 12))
        _marks[idx] = 1;
    else
        _marks[idx] = 0;
    if (idx == 0)
        _marks[numParticle] = 1;
}

__global__ void buildHashMapFromPage(const int numPage, const int tableSize, const unsigned long long* d_masks, const unsigned long long* d_particleOffsets,
    const int* d_page2particle, unsigned long long* d_keyTable, int* d_valTable, unsigned long long* d_pageOffsets) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= numPage) return;
    unsigned long long key = d_particleOffsets[d_page2particle[idx]] >> 12;
    unsigned long long hashkey = key % tableSize;

    d_pageOffsets[idx] = key << 12;
    do {
        atomicCAS((unsigned long long int*)d_keyTable + hashkey, 0xffffffffffffffff, (unsigned long long int)key);    ///< -1 is the default value, means unoccupied
        if (d_keyTable[hashkey] == key) {  ///< haven't found related record, so create a new entry
            d_valTable[hashkey] = idx;   ///< created a record
            break;
        }
        else {
            hashkey += 127; ///< search next entry
            if (hashkey >= tableSize) hashkey = hashkey % tableSize;
        }
    } while (true);
}

__global__ void supplementAdjacentPages(const int numPage, const int tableSize, const unsigned long long* d_masks,
    const unsigned long long* d_particleOffsets, const unsigned long long* d_neighborOffsets, const int* d_page2particle,
    unsigned long long* d_keyTable, int* d_valTable, int* d_totalPage, unsigned long long* d_pageOffsets) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= numPage) return;

    int sparsePageId;
    unsigned long long okey = d_particleOffsets[d_page2particle[idx]] & 0xfffffffffffff000;
    for (int i = 0; i < 7; i++) {
        unsigned long long key = Packed_Add(d_masks, okey, d_neighborOffsets[i]) >> 12;    ///< dense page id, used as key
        unsigned long long hashkey = key % tableSize;
        while (d_keyTable[hashkey] != key) {
            unsigned long long old = atomicCAS((unsigned long long int*)d_keyTable + hashkey, 0xffffffffffffffff, (unsigned long long int)key);    ///< -1 is the default value, means unoccupied
            if (d_keyTable[hashkey] == key) {
                if (old == 0xffffffffffffffff) {  ///< created a new entry
                    d_valTable[hashkey] = sparsePageId = atomicAdd(d_totalPage, 1);   ///< created a record
                    d_pageOffsets[sparsePageId] = key << 12;
                }
                break;
            }
            else {
                hashkey += 127; ///< search next entry
                if (hashkey >= tableSize) hashkey = hashkey % tableSize;
            }
        }
    }
}

__global__ void establishPageTopology(const int numPage, const int tableSize, const unsigned long long* d_masks,
    const unsigned long long* d_particleOffsets, const unsigned long long* d_neighborOffsets, const int* d_page2particle,
    unsigned long long* d_keyTable, int* d_valTable, int** d_adjPage) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= numPage) return;

    unsigned long long okey = d_particleOffsets[d_page2particle[idx]] & 0xfffffffffffff000;
    for (int i = 0; i < 7; i++) {
        unsigned long long key = Packed_Add(d_masks, okey, d_neighborOffsets[i]) >> 12;    ///< dense page id, used as key
        unsigned long long hashkey = key % tableSize;
        while (d_keyTable[hashkey] != key) {
            hashkey += 127; ///< search next entry
            if (hashkey >= tableSize) hashkey = hashkey % tableSize;
        }
        d_adjPage[i][idx] = d_valTable[hashkey];
    }
}

__global__ void markPageSize(const int numPage, const int* _offsets, int* _marks) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numPage) return;
    _marks[idx] = (_offsets[idx + 1] - _offsets[idx] + 255) / 256;
}

__global__ void markVirtualPageOffset(const int numPage, const int* _toVirtualOffsets, int* _marks) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numPage) return;
    _marks[_toVirtualOffsets[idx]] = 1;
}


void  DomainTransformer::establishParticleGridMapping() {
    // in order to calc cell num
    const unsigned int threadNum = 256;
    const unsigned int blockNum = (_numParticle + threadNum - 1) / threadNum;


    CUDA_SAFE_CALL(cudaMemset(d_marks, 0, sizeof(int) * _numParticle));
    markCellBoundary << <blockNum, threadNum >> > (_numParticle, d_offsets, d_marks);

	if (_numParticle < pretype_threshold) {
		In_Prefix_Sum_Int(d_marks, d_particle2cell, _numParticle);
	}
	else {
		(thrust::inclusive_scan(thrust::device_ptr<int>(d_marks), thrust::device_ptr<int>(d_marks) + _numParticle, thrust::device_ptr<int>(d_particle2cell)));
	}
    CUDA_SAFE_CALL(cudaMemcpy(&_numCell, (d_particle2cell)+_numParticle - 1, sizeof(int), cudaMemcpyDeviceToHost));

    markBlockOffset << <blockNum, threadNum >> > (_numParticle, d_particle2cell, d_cell2particle);
   
    // page-based
    CUDA_SAFE_CALL(cudaMemset((d_marks), 0, sizeof(int) * _numParticle));

    markPageBoundary << <blockNum, threadNum >> > (_numParticle, d_offsets, d_marks);
	if (_numParticle < pretype_threshold) {
		In_Prefix_Sum_Int(d_marks, d_particle2page, _numParticle);
	}
	else {
		(thrust::inclusive_scan(thrust::device_ptr<int>(d_marks), thrust::device_ptr<int>(d_marks) + _numParticle, thrust::device_ptr<int>(d_particle2page)));
	}

	CUDA_SAFE_CALL(cudaMemcpy(&_numPage, (d_particle2page)+_numParticle - 1, sizeof(int), cudaMemcpyDeviceToHost));
    
    markBlockOffset << <blockNum, threadNum >> > (_numParticle, d_particle2page, d_page2particle);
}

void  DomainTransformer::establishHashmap() {
    CUDA_SAFE_CALL(cudaMemset(d_keyTable, 0xffffffffffffffff, sizeof(unsigned long long) * _tableSize));
    CUDA_SAFE_CALL(cudaMemset(d_valTable, 0xffffffffffffffff, sizeof(int) * _tableSize));

    const unsigned int threadNum = 512;
    const unsigned int blockNum = (_numPage + threadNum - 1) / threadNum;


    buildHashMapFromPage << <blockNum, threadNum >> > (_numPage, _tableSize, d_masks, d_offsets, (const int*)(d_page2particle), d_keyTable, d_valTable, d_pageOffset);
    
    CUDA_SAFE_CALL(cudaMemcpy(d_totalPage, &_numPage, sizeof(int), cudaMemcpyHostToDevice));

    supplementAdjacentPages << <blockNum, threadNum >> > (_numPage, _tableSize, d_masks, d_offsets, (const unsigned long long*)d_neighborOffsets, (const int*)(d_page2particle), d_keyTable, d_valTable,
        d_totalPage, d_pageOffset);
    establishPageTopology << <blockNum, threadNum >> > (_numPage, _tableSize, d_masks, d_offsets, (const unsigned long long*)d_neighborOffsets, (const int*)(d_page2particle), d_keyTable, d_valTable,
        d_adjPage);

    CUDA_SAFE_CALL(cudaMemcpy(&_numTotalPage, d_totalPage, sizeof(int), cudaMemcpyDeviceToHost));
}

void  DomainTransformer::buildTargetPage() {

    const unsigned int threadNum = 512;
    const unsigned int blockNum = (_numPage + threadNum - 1) / threadNum;

    markPageSize << <blockNum, threadNum >> > (_numPage, (const int*)d_page2particle, d_marks);
    
	if (_numPage < pretype_threshold) {
		Ex_Prefix_Sum_Int(d_marks, d_virtualPageOffset, _numPage + 1);
	}
	else {
		(thrust::exclusive_scan(thrust::device_ptr<int>(d_marks), thrust::device_ptr<int>(d_marks) + _numPage + 1, thrust::device_ptr<int>(d_virtualPageOffset)));
	}

    CUDA_SAFE_CALL(cudaMemcpy(&_numVirtualPage, d_virtualPageOffset + _numPage, sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_SAFE_CALL(cudaMemset(d_marks, 0, sizeof(int) * _numVirtualPage));

    markVirtualPageOffset << <blockNum, threadNum >> > (_numPage, (const int*)d_virtualPageOffset, d_marks);
    
	if (_numVirtualPage < pretype_threshold) {
		In_Prefix_Sum_Int(d_marks, d_targetPage, _numVirtualPage);
	}
	else {
		(thrust::inclusive_scan(thrust::device_ptr<int>(d_marks), thrust::device_ptr<int>(d_marks) + _numVirtualPage, thrust::device_ptr<int>(d_targetPage)));
	}
}