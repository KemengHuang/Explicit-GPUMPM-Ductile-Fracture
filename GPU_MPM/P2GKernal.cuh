#ifndef __P2G_KERNAL_CUH_
#define __P2G_KERNAL_CUH_
#include <cuda_runtime.h>
//#include <stdint.h>
#include "Setting.h"



    //__global__ void P2G_FLIP(
    //    const int numParticle, const int* d_targetPages, const int* d_virtualPageOffsets,
    //    const int3* smallest_nodes, const T* d_sigma,
    //    int* d_block_offsets, int* d_cellids, int* d_indices, int* d_indexTrans,
    //    vector3T* d_sorted_positions, T* d_sorted_masses, vector3T* d_sorted_velocities, T** d_channels,
    //    int** d_adjPage);

    __global__ void P2G_APIC(
        const int numParticle, const int* d_targetPages, const int* d_virtualPageOffsets,
        const int3* smallest_nodes, const T* d_sigma,
        int* d_block_offsets, int* d_cellids, int* d_indices, int* d_indexTrans,
        vector3T* d_sorted_positions, T* d_sorted_C, T* d_sorted_masses, vector3T* d_sorted_velocities, T* d_B, T** d_channels,
        int** d_adjPage);

    __global__ void P2G_APIC_CONFLICT_FREE(
        const int numParticle, const int* d_targetPages, const int* d_virtualPageOffsets,
        const int3* smallest_nodes, const T* d_sigma,
        int* d_block_offsets, int* d_cellids, int* d_indices, int* d_indexTrans,
        vector3T* d_sorted_positions, T* d_sorted_masses, vector3T* d_sorted_velocities, T* d_B, T** d_channels,
        int** d_adjPage);

    __global__ void P2G_MLS(
        const int numParticle, const int* d_targetPages, const int* d_virtualPageOffsets,
        const int3* smallest_nodes, const T* d_sigma,
        int* d_block_offsets, int* d_cellids, int* d_indices, int* d_indexTrans,
        vector3T* d_sorted_positions, T* d_sorted_C, T* d_sorted_masses, vector3T* d_sorted_velocities, T* d_B, const T dt,
        T** d_channels, int** d_adjPage, unsigned long long* d_pageOffsets);

    __global__ void volP2G_APIC(
        const int numParticle, const int* d_targetPages, const int* d_virtualPageOffsets,
        const int3* smallest_nodes, 
        int* d_block_offsets, int* d_cellids, int* d_indices, int* d_indexTrans,
        vector3T* d_sorted_positions, T* d_sorted_vol,  T** d_channels,
        int** d_adjPage, 
        T dt,
        T parabolic_M);

    __global__ void preConditionP2G_APIC(
        const int numParticle, const int* d_targetPages, const int* d_virtualPageOffsets,
        const int3* smallest_nodes,
        int* d_block_offsets, int* d_cellids, int* d_indices, int* d_indexTrans,
        vector3T* d_sorted_positions, T* d_sorted_vol, T* d_sorted_FP, T** d_channels,
        int** d_adjPage,
        T dt,
        T parabolic_M);

    __global__ void AxP2G_APIC(
        const int numParticle,
        const int* d_targetPages,
        const int* d_virtualPageOffsets,
        const int3* smallest_nodes,
        int* d_block_offsets,
        int* d_cellids,
        int* d_indices,
        int* d_indexTrans,
        vector3T* d_sorted_positions,
        vector3T* d_sorted_col,
        T* d_sorted_vol,
        T* d_ax,
        T* d_FP,
        T* d_channels,
        int** d_adjPage,
        T dt,
        T parabolic_M);

#endif
