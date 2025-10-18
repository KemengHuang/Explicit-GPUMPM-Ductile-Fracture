#pragma once
#include "Setting.h"
class SPGrid {   //: public AttribConnector<static_cast<int>(ParticleAttribIndex::NUM_ATTRIBS), 1> {
public:
    //friend class MPMSimulator;
    SPGrid() {};
    void SPGrid_MEM_Malloc();
    void SPGrid_MEM_Free();
    void initialize(unsigned long long* dMasks, unsigned long long* hMasks, int weight, int height, int depth, T dc);
    ~SPGrid();
    void clear();
    

public:
    /// input
    int _width, _height, _depth;
    T _memoryScale;
    T   _dc;
    unsigned long long* d_masks;
    unsigned long long* h_masks;

    // grid data
    T* hd_channels[15];
    T* d_grid;
    T* d_grid_r;
    T* d_grid_mr;
    T* d_grid_s;
    T* d_grid_q;
    T* d_grid_x;
    T* d_grid_temp;
	T* _maxVel;
	T* innerProductR;
    T** d_channels;
    //unsigned* d_flags;

};
