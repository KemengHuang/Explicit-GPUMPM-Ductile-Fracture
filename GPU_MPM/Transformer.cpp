#include "Transformer.cuh"
#include"cuda_tools.h"
#include"SPGridMask.h"

void DomainTransformer::Transformer_Malloc_MEM(){
	
	const unsigned int trans_array_size = _numParticle + 2;
	
	CUDA_SAFE_CALL(cudaMalloc((void**)&d_marks, sizeof(int) * trans_array_size));
	CUDA_SAFE_CALL(cudaMalloc((void**)&d_particle2cell, sizeof(int) * trans_array_size));
	CUDA_SAFE_CALL(cudaMalloc((void**)&d_cell2particle, sizeof(int) * trans_array_size));
	CUDA_SAFE_CALL(cudaMalloc((void**)&d_particle2page, sizeof(int) * trans_array_size));
	CUDA_SAFE_CALL(cudaMalloc((void**)&d_page2particle, sizeof(int) * trans_array_size));
	CUDA_SAFE_CALL(cudaMalloc((void**)&d_virtualPageOffset, sizeof(int) * trans_array_size));

	cudaMemset(d_virtualPageOffset, 0, sizeof(int));
	CUDA_SAFE_CALL(cudaMalloc((void**)&d_targetPage, sizeof(int) * trans_array_size));
	CUDA_SAFE_CALL(cudaMalloc((void**)&d_totalPage, sizeof(int)));

	for (auto& adjpage : hd_adjPage)
		CUDA_SAFE_CALL(cudaMalloc((void**)&adjpage, sizeof(int) * (int)(_gridVolume * MEMORY_SCALE / 4 / 4 / 4)));
	CUDA_SAFE_CALL(cudaMalloc((void**)&d_adjPage, sizeof(int*) * 7));
	CUDA_SAFE_CALL(cudaMemcpy(d_adjPage, hd_adjPage, sizeof(int*) * 7, cudaMemcpyHostToDevice));

	CUDA_SAFE_CALL(cudaMalloc((void**)&d_pageOffset, sizeof(unsigned long long) * (int)(_gridVolume * MEMORY_SCALE / 4 / 4 / 4)));

	unsigned long long h_neighborOffsets[8];
	
	for (int i = 0; i < 2; ++i)
		for (int j = 0; j < 2; ++j)
			for (int k = 0; k < 2; ++k)
				h_neighborOffsets[i * 4 + j * 2 + k] = SPGridMask::Linear_Offset(h_masks, i * 4, j * 4, k * 4);
	CUDA_SAFE_CALL(cudaMalloc((void**)&d_neighborOffsets, sizeof(unsigned long long) * 7));
	CUDA_SAFE_CALL(cudaMemcpy(d_neighborOffsets, h_neighborOffsets + 1, sizeof(unsigned long long) * 7, cudaMemcpyHostToDevice));
}

void DomainTransformer::initialize(const int numParticle, const int gridVolume, const unsigned long long* dMasks, unsigned long long* hMasks, const unsigned long long* dOffsets, const int tableSize, unsigned long long* keyTable, int* valTable) {
	_numParticle = numParticle;
	_gridVolume = gridVolume;
	h_masks = hMasks;
	d_masks = dMasks;
	d_offsets = dOffsets;
	_tableSize = tableSize;
	d_keyTable = keyTable;
	d_valTable = valTable;
	
}


DomainTransformer::~DomainTransformer() {
	CUDA_SAFE_CALL(cudaFree(d_pageOffset));
	CUDA_SAFE_CALL(cudaFree(d_neighborOffsets));
	CUDA_SAFE_CALL(cudaFree(d_adjPage));
	for (auto& adjpage : hd_adjPage)
		CUDA_SAFE_CALL(cudaFree(adjpage));
	CUDA_SAFE_CALL(cudaFree(d_totalPage));


	CUDA_SAFE_CALL(cudaFree(d_marks));
	CUDA_SAFE_CALL(cudaFree(d_particle2cell));
	CUDA_SAFE_CALL(cudaFree(d_cell2particle));
	CUDA_SAFE_CALL(cudaFree(d_particle2page));
	CUDA_SAFE_CALL(cudaFree(d_page2particle));
	CUDA_SAFE_CALL(cudaFree(d_virtualPageOffset));
	CUDA_SAFE_CALL(cudaFree(d_targetPage));
}




void DomainTransformer::rebuild() {
	// clean before computing
	CUDA_SAFE_CALL(cudaMemset((void*)d_pageOffset, 0, sizeof(unsigned long long) * (int)(_gridVolume * MEMORY_SCALE / 4 / 4 / 4)));
	for (auto& adjpage : hd_adjPage)
		CUDA_SAFE_CALL(cudaMemset((void*)adjpage, 0, sizeof(int) * (int)(_gridVolume * MEMORY_SCALE / 4 / 4 / 4)));

	establishParticleGridMapping();
	establishHashmap();
	CUDA_SAFE_CALL(cudaMemcpy(&_numTotalPage, d_totalPage, sizeof(int), cudaMemcpyDeviceToHost));
	buildTargetPage();
	//printf("_numTotalPage is %d \n", _numTotalPage);
}



