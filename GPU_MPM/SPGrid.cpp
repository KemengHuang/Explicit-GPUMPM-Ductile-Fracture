#include "SPGrid.h"
#include"cuda_tools.h"
#include "SPGridMask.h"

template<class T, class T_FIELD> unsigned long long
OffsetOfMember(T_FIELD T::* field){
    return (unsigned long long)((char*)&(((T*)0)->*field) - (char*)0);
}

void SPGrid::SPGrid_MEM_Malloc() {

    
    //SPGridMask spgdm(Log2(sizeof(CH_STRUCT)), Log2(sizeof(T)));

    _memoryScale = MEMORY_SCALE;
    unsigned long long int nodeSize = _width * _height * _depth;
    unsigned long long int tmp = sizeof(CH_STRUCT) * nodeSize * _memoryScale;
    unsigned long long int tmpT = sizeof(T) * nodeSize * _memoryScale;
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_grid, tmp)); // more secured way should be used
   
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_grid_temp, tmpT));
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_grid_r, tmpT));
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_grid_mr, tmpT));
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_grid_s, tmpT));
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_grid_q, tmpT));
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_grid_x, tmpT));
	CUDA_SAFE_CALL(cudaMalloc((void**)&innerProductR, sizeof(T)));
	CUDA_SAFE_CALL(cudaMalloc((void**)&_maxVel, sizeof(T)));

    CUDA_SAFE_CALL(cudaMalloc((void**)&d_channels, sizeof(T*) * 15));

    printf("size one struct(%u Bytes) total(%llu Bytes)\n", (unsigned int)sizeof(CH_STRUCT), tmp);

    unsigned int elements_per_block = 64;
    hd_channels[0] = reinterpret_cast<T*>((unsigned long long)d_grid + unsigned long long(OffsetOfMember(&CH_STRUCT::ch0) * elements_per_block)); // mass
    hd_channels[1] = reinterpret_cast<T*>((unsigned long long)d_grid + unsigned long long(OffsetOfMember(&CH_STRUCT::ch1) * elements_per_block)); // vel momentum
    hd_channels[2] = reinterpret_cast<T*>((unsigned long long)d_grid + unsigned long long(OffsetOfMember(&CH_STRUCT::ch2) * elements_per_block)); // vel
    hd_channels[3] = reinterpret_cast<T*>((unsigned long long)d_grid + unsigned long long(OffsetOfMember(&CH_STRUCT::ch3) * elements_per_block)); // vel
    hd_channels[4] = reinterpret_cast<T*>((unsigned long long)d_grid + unsigned long long(OffsetOfMember(&CH_STRUCT::ch4) * elements_per_block)); // force
    hd_channels[5] = reinterpret_cast<T*>((unsigned long long)d_grid + unsigned long long(OffsetOfMember(&CH_STRUCT::ch5) * elements_per_block)); // force
    hd_channels[6] = reinterpret_cast<T*>((unsigned long long)d_grid + unsigned long long(OffsetOfMember(&CH_STRUCT::ch6) * elements_per_block)); // force
    hd_channels[7] = reinterpret_cast<T*>((unsigned long long)d_grid + unsigned long long(OffsetOfMember(&CH_STRUCT::ch7) * elements_per_block)); // 
    hd_channels[8] = reinterpret_cast<T*>((unsigned long long)d_grid + unsigned long long(OffsetOfMember(&CH_STRUCT::ch8) * elements_per_block)); // 
    hd_channels[9] = reinterpret_cast<T*>((unsigned long long)d_grid + unsigned long long(OffsetOfMember(&CH_STRUCT::ch9) * elements_per_block)); // 
    hd_channels[10] = reinterpret_cast<T*>((unsigned long long)d_grid + unsigned long long(OffsetOfMember(&CH_STRUCT::ch10) * elements_per_block)); // 
    hd_channels[11] = reinterpret_cast<T*>((unsigned long long)d_grid + unsigned long long(OffsetOfMember(&CH_STRUCT::ch11) * elements_per_block)); // 
    hd_channels[12] = reinterpret_cast<T*>((unsigned long long)d_grid + unsigned long long(OffsetOfMember(&CH_STRUCT::ch12) * elements_per_block)); // 
    hd_channels[13] = reinterpret_cast<T*>((unsigned long long)d_grid + unsigned long long(OffsetOfMember(&CH_STRUCT::ch13) * elements_per_block)); // 
    hd_channels[14] = reinterpret_cast<T*>((unsigned long long)d_grid + unsigned long long(OffsetOfMember(&CH_STRUCT::ch14) * elements_per_block)); // 
  
    CUDA_SAFE_CALL(cudaMemcpy(d_channels, hd_channels, sizeof(T*) * 15, cudaMemcpyHostToDevice));

    //d_flags = reinterpret_cast<unsigned*>((unsigned long long)d_grid + unsigned long long(OffsetOfMember(&CH_STRUCT::flags) * elements_per_block));
    
}

void SPGrid::initialize(unsigned long long* dMasks, unsigned long long* hMasks, int width, int height, int depth, T dc) {
    _width = width; _height = height; _depth = depth; _dc = dc;
    d_masks = dMasks;
    h_masks = hMasks;
}

SPGrid::~SPGrid() {
    CUDA_SAFE_CALL(cudaFree(d_grid));
    CUDA_SAFE_CALL(cudaFree(d_channels)); 
    SPGrid_MEM_Free();
}
void SPGrid::SPGrid_MEM_Free() {
    CUDA_SAFE_CALL(cudaFree(d_grid));
    CUDA_SAFE_CALL(cudaFree(d_channels));
    CUDA_SAFE_CALL(cudaFree(d_grid_temp));
    CUDA_SAFE_CALL(cudaFree(d_grid_r));
    CUDA_SAFE_CALL(cudaFree(d_grid_mr));
    CUDA_SAFE_CALL(cudaFree(d_grid_s));
    CUDA_SAFE_CALL(cudaFree(d_grid_q));
    CUDA_SAFE_CALL(cudaFree(d_grid_x));
	CUDA_SAFE_CALL(cudaFree(innerProductR));
	CUDA_SAFE_CALL(cudaFree(_maxVel));
}
void SPGrid::clear() {
    unsigned long long int totalSize = sizeof(CH_STRUCT) * _width * _height * _depth * _memoryScale;
    CUDA_SAFE_CALL(cudaMemset(d_grid, 0, totalSize));
}

