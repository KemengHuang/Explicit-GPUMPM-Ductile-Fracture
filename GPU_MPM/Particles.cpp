#include"Particles.cuh"
#include"cuda_tools.h"
#include <sstream>
#include<string>
#include"fstream"
Particles::Particles() {}

void Particles::initialize(unsigned int numParticle, int tableSize, const T* h_mass, const vector3T* h_pos, vector3T* h_vel, unsigned long long* dp_dmasks, unsigned long long* keyTable, int* valTable, T* _memTrunk, int* pageId, int* pageoffset) {
    _numParticle = numParticle;
    _tableSize = tableSize;
    Particles_Malloc_MEM();
	
	

    d_memTrunk = _memTrunk;
    d_keyTable = keyTable;  ///< offset (morton code)
    d_valTable = valTable;  ///< sparse page id
    d_pageId = pageId;
	d_page_offset = pageoffset;
    _dmasks = dp_dmasks;

    for (int i = 0; i < numParticle; i++) {
        h_color[i] = (static_cast<unsigned int>((1) * 255) << 24) |
            (static_cast<unsigned int>((0.5) * 255.0f) << 16) |
            (static_cast<unsigned int>((0.5) * 255.0f) << 8) |
            static_cast<unsigned int>((0.0) * 255.0f);
    }
    CUDA_SAFE_CALL(cudaMemcpy(d_color, h_color, sizeof(unsigned int) * _numParticle, cudaMemcpyHostToDevice));

    initialize_kernal(h_mass, h_pos, h_vel);
}
int iii = 0;
void Particles::Attribs_Back_Host(std::vector<vector3T>& h_pos, std::vector<vector3T>& h_vel, std::vector<unsigned int>& h_indices) {

    CUDA_SAFE_CALL(cudaMemcpy(&h_pos[0], d_pos, sizeof(vector3T) * _numParticle, cudaMemcpyDeviceToHost));
    CUDA_SAFE_CALL(cudaMemcpy(_phaseC_trans, d_phase_C, sizeof(T) * _numParticle, cudaMemcpyDeviceToHost));

    //if (iii > 200) {
    //    //std::stringstream ss;
    //    //ss << "phase " << iii << ".txt";
    //    //std::string filename;
    //    //ss >> filename;
    //    //std::ofstream out(filename);
    //    for (int i = 0; i < _numParticle; i++) {
    //        if (_phaseC_trans[i] >= 0.0f && _phaseC_trans[i] <= 1.0f) //continue;
    //            std::cout << _phaseC_trans[i] << "    ";// std::endl;
    //    }
    //    
    //}
    //iii++;
    CUDA_SAFE_CALL(cudaMemcpy(h_color, d_color, sizeof(unsigned int) * _numParticle, cudaMemcpyDeviceToHost));
}

void Particles::Particles_Malloc_MEM() {
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_indices, sizeof(int) * _numParticle));
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_indexTrans, sizeof(int) * _numParticle));
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_numBucket, sizeof(int)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_particle2bucket, sizeof(int) * _numParticle));
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_bucketSizes, sizeof(int) * (_tableSize<<6)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_bucketOffsets, sizeof(int) * (_tableSize<<6)));

    cudaMemset(d_bucketOffsets, 0, sizeof(int));

    /// alias 
    const unsigned int particle_array_size = _numParticle + 2;

    CUDA_SAFE_CALL(cudaMalloc((void**)&cell_id, sizeof(unsigned long long) * particle_array_size));
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_mass, sizeof(T) * particle_array_size));

    CUDA_SAFE_CALL(cudaMalloc((void**)&d_orderedMass, sizeof(T) * particle_array_size));
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_pos, sizeof(vector3T) * particle_array_size));
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_orderedVel, sizeof(vector3T) * particle_array_size));
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_orderedCol, sizeof(vector3T) * particle_array_size));
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_orderedPos, sizeof(vector3T) * particle_array_size));
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_smallestNodeIndex, sizeof(int3) * particle_array_size));
    CUDA_SAFE_CALL(cudaMalloc((void**)&velSqu, sizeof(T) * particle_array_size));

    _phaseC_trans = new T[_numParticle];
    h_color = new unsigned int[_numParticle];
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_phase_C, sizeof(T) * particle_array_size));
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_phase_C_sort, sizeof(T) * particle_array_size));
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_maxPsi, sizeof(T) * particle_array_size));
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_color, sizeof(unsigned int) * particle_array_size));

    CUDA_SAFE_CALL(cudaMalloc((void**)&d_vol, sizeof(T) * _numParticle));
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_FP, sizeof(T) * _numParticle));

    CUDA_SAFE_CALL(cudaMalloc((void**)&d_g, sizeof(T) * _numParticle));
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_alpha, sizeof(T) * _numParticle));

    CUDA_SAFE_CALL(cudaMalloc((void**)&d_min, sizeof(vector3T)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_F, sizeof(T) * 9 * _numParticle));
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_B, sizeof(T) * 9 * _numParticle));
}

Particles::~Particles() {
    CUDA_SAFE_CALL(cudaFree(d_min));
    CUDA_SAFE_CALL(cudaFree(d_smallestNodeIndex));
    CUDA_SAFE_CALL(cudaFree(d_orderedVel));
    CUDA_SAFE_CALL(cudaFree(d_orderedPos));
    CUDA_SAFE_CALL(cudaFree(d_orderedCol));
    CUDA_SAFE_CALL(cudaFree(d_pos));
    CUDA_SAFE_CALL(cudaFree(d_indices));
    CUDA_SAFE_CALL(cudaFree(d_indexTrans));
    CUDA_SAFE_CALL(cudaFree(d_F));
    CUDA_SAFE_CALL(cudaFree(d_B));
    CUDA_SAFE_CALL(cudaFree(d_mass));
    CUDA_SAFE_CALL(cudaFree(d_orderedMass));
    CUDA_SAFE_CALL(cudaFree(d_numBucket));
    CUDA_SAFE_CALL(cudaFree(d_particle2bucket));
    CUDA_SAFE_CALL(cudaFree(d_bucketSizes));
    CUDA_SAFE_CALL(cudaFree(d_bucketOffsets));
    CUDA_SAFE_CALL(cudaFree(cell_id));
    CUDA_SAFE_CALL(cudaFree(velSqu));
    delete[] h_color;
    delete[] _phaseC_trans;
    CUDA_SAFE_CALL(cudaFree(d_phase_C));
    CUDA_SAFE_CALL(cudaFree(d_phase_C_sort));
    CUDA_SAFE_CALL(cudaFree(d_maxPsi));
    CUDA_SAFE_CALL(cudaFree(d_color));
    CUDA_SAFE_CALL(cudaFree(d_vol));
    CUDA_SAFE_CALL(cudaFree(d_FP));
    CUDA_SAFE_CALL(cudaFree(d_alpha));
    CUDA_SAFE_CALL(cudaFree(d_g));
	//CUDA_SAFE_CALL(cudaFree(recursiveSum));
}