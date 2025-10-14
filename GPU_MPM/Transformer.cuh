#pragma once
#ifndef __TRANSFORMER_H_
#define __TRANSFORMER_H_
//#include <cuda_runtime.h>
//#include <thrust/device_vector.h>

#include "Setting.h"

class DomainTransformer {
public:
    //friend class MPMSimulator;
    void  establishParticleGridMapping();
    void  establishHashmap();
    void  buildTargetPage();

    DomainTransformer() {};
    void Transformer_Malloc_MEM();
    void initialize(const int numParticle, const int gridVolume, const unsigned long long* dMasks, unsigned long long* hMasks, const unsigned long long* dOffsets, const int tableSize, unsigned long long* keyTable, int* valTable);

    ~DomainTransformer();
    void  rebuild();

public:
    /// input
    int                 _numParticle;
    int                 _gridVolume;
    int                 _tableSize;
    const unsigned long long* d_masks;
    unsigned long long*  h_masks;
    const unsigned long long* d_offsets;
    unsigned long long* d_keyTable;  ///< offset (morton code)
    int* d_valTable;  ///< sparse page id
/// hash
    int* d_totalPage;
    int* hd_adjPage[7];
    int** d_adjPage;
    unsigned long long* d_pageOffset;

    int                 _numPage, _numCell;
    int                 _numTotalPage, _numVirtualPage;
    unsigned long long* d_neighborOffsets;
    /// mapping
    int* d_marks;
    int* d_particle2page;	///< particle no. -> cell no.
    int* d_page2particle;	///< cell no. -> particle offset
    int* d_particle2cell;
    int* d_cell2particle;
    //int*          d_pageSize;       ///< number of particles (or blocks needed) per page, replaced by d_marks
    int* d_virtualPageOffset;
    int* d_targetPage;     ///< virtual page no. -> actual (target) page no.
};

#endif

