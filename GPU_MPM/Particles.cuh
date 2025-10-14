#pragma once
#ifndef __PARTICLE_DOMAIN_H_
#define __PARTICLE_DOMAIN_H_
#include <cuda_runtime.h>
#include "Setting.h"
#include<vector>
class Particles {
public:
    //friend class MPMSimulator;
    Particles();
    void Particles_Malloc_MEM();
    ~Particles();
    void	initialize(unsigned int numParticle, int tableSize, const T* h_mass, const vector3T* h_pos, vector3T* h_vel, unsigned long long* _dmasks, unsigned long long* keyTable, int* valTable, T* _memTrunk, int* pageId, int* pageoffset);
    void    initialize_kernal(const T* h_mass, const vector3T* h_pos, vector3T* h_vel);
    void    sort_by_offsets(const unsigned int& blockNum, const unsigned int threadNum);
    void	reorder();
    void    getMaxParticleVel(T* maxVelSqared);
	void    getMaxParticleVel_c(T* maxVelSqared);
    void    getMaxParticleVel_b(T* maxVelSqared);
    void Attribs_Back_Host(std::vector<vector3T>& h_pos, std::vector<vector3T>& h_vel, std::vector<unsigned int>& h_indices);
public:
    /// input
    unsigned int     _numParticle;
    vector3T* d_min;
    vector3T    _min;

    /// attribs
    unsigned long long* cell_id;
    T* d_mass;
    T* d_orderedMass;

    vector3T* d_pos;
    vector3T* d_orderedPos;
    vector3T* d_orderedCol;
    vector3T* d_orderedVel;
    int3* d_smallestNodeIndex;

    T* velSqu;


    /// auxiliary
    int* d_indices;
    int* d_indexTrans;

    int _numBucket;
    int* d_numBucket;
    int* d_particle2bucket;
    int* d_bucketSizes;
    int* d_bucketOffsets;

    int   _tableSize;
    T* d_F;
    T* d_B;
    T* d_vol;
    T* d_FP;

    T* d_memTrunk;
    unsigned long long* d_keyTable;  ///< offset (morton code)
    int* d_valTable;  ///< sparse page id
    int* d_pageId;
	int* d_page_offset;
    unsigned long long* _dmasks;


    T* _phaseC_trans;
    T* d_phase_C;
    T* d_phase_C_sort;
    T* d_maxPsi;

    T* d_g;
    T* d_alpha;

    unsigned int* h_color;
    unsigned int* d_color;
};

#endif
