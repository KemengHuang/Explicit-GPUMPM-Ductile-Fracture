#ifndef __G2P_KERNAL_CUH_
#define __G2P_KERNAL_CUH_
#include <cuda_runtime.h>
#include <stdint.h>
#include "Setting.h"



    //__global__ void G2P_FLIP(
    //    const int numParticle, const int* d_targetPages, const int* d_virtualPageOffsets, const int3* smallest_nodes,
    //    int* d_block_offsets, int* d_cellids, int* d_indices, int* d_indexTrans, vector3T* d_sorted_positions, vector3T* d_sorted_velocities,
    //    T** d_channels, T* d_sorted_F, T* d_tmp, T dt, int** d_adjPage);

    __global__ void G2P_APIC(
        const int numParticle, const int* d_targetPages, const int* d_virtualPageOffsets, const int3* smallest_nodes,
        int* d_block_offsets, int* d_cellids, int* d_indices, int* d_indexTrans, vector3T* d_sorted_positions, T* d_sorted_C, T* d_sorted_C_sort, vector3T* d_sorted_velocities,
        T** d_channels, T* d_sorted_F, T* d_sorted_B, T* d_tmp, T dt, int** d_adjPage);

    __global__ void G2P_APIC_CONFLICT_FREE(
        const int numParticle, const int* d_targetPages, const int* d_virtualPageOffsets, const int3* smallest_nodes,
        int* d_block_offsets, int* d_cellids, int* d_indices, int* d_indexTrans, vector3T* d_sorted_positions, vector3T* d_sorted_velocities,
        T** d_channels, T* d_sorted_F, T* d_sorted_B, T* d_tmp, T dt, int** d_adjPage);

    __global__ void G2P_MLS(
        const int numParticle, const int* d_targetPages, const int* d_virtualPageOffsets, const int3* smallest_nodes,
        int* d_block_offsets, int* d_cellids, int* d_indices, int* d_indexTrans, vector3T* d_sorted_positions, T* d_sorted_C,
		T* d_sorted_C_sort, vector3T* d_sorted_velocities,
        T** d_channels, T* d_sorted_F, T* d_sorted_B, T* d_tmp, T dt, int** d_adjPage);

    __global__ void AxG2P_APIC(
        const int numParticle,
        const T L0,
        const int* d_targetPages,
        const int* d_virtualPageOffsets,
        const int3* smallest_nodes,
        int* d_block_offsets,
        int* d_cellids,
        int* d_indices,
        int* d_indexTrans,
        vector3T* d_sorted_positions,
        T* d_sorted_vol,
        vector3T* d_sorted_col,
        T* d_channels,
        int** d_adjPage);


#endif
