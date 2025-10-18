#include"TimeIntegrator.cuh"
#include"cuda_tools.h"
#include"P2GKernal.cuh"
#include"G2PKernal.cuh"
//__device__ int Bit_Pack_Mine(const unsigned long long mask, const unsigned long long data) {
//    union { unsigned long long slresult; unsigned long long ulresult; };
//    unsigned long long uldata = data; int count = 0; ulresult = 0;
//
//    unsigned long long rmask = __brevll(mask);
//    unsigned char lz;
//
//    while (rmask) {
//        lz = __clzll(rmask);
//        uldata >>= lz;
//        ulresult <<= 1;
//        count++;
//        ulresult |= (uldata & 1);
//        uldata >>= 1;
//        rmask <<= lz + 1;
//    }
//    ulresult <<= 64 - count; // 64 means 64 bits ... maybe not use a constant 64 ...
//    ulresult = __brevll(ulresult);
//    return (int)slresult;
//}

__device__ int Bit_Pack_Mine(const unsigned long long& mask, const unsigned long long& data) {
    unsigned long long uldata = data; int count = 0; unsigned long long ulresult = 0;
    unsigned long long rmask = __brevll(mask);
    unsigned char lz;

    while (rmask && uldata) {
        lz = __clzll(rmask);
        uldata >>= lz;
        ulresult <<= 1;

        ulresult |= (uldata & 1);
        count++;
        uldata >>= 1;
        rmask <<= lz + 1;
        //count += (lz + 1);
    }
    //ulresult <<= 64 - count; // 64 means 64 bits ... maybe not use a constant 64 ...
    ulresult = __brevll(ulresult) >> (64 - count);
    return (int)ulresult;
}
__global__ void initMatrix_kernal(const int numParticle, T* d_matrix) {
    int parid = blockDim.x * blockIdx.x + threadIdx.x;
    if (parid >= numParticle) return;
    for (int i = 0; i < 9; i++)
        d_matrix[parid + i * numParticle] = ((i % 3) == (i / 3));
}

__global__ void calcIndex(
    const int numCell, const T one_over_dx, const int* d_cell_first_particles_indices,
    const vector3T* d_sorted_positions, int3* smallest_nodes) {
    int cellid = blockDim.x * blockIdx.x + threadIdx.x;
    if (cellid >= numCell) return;
    smallest_nodes[cellid].x = (int)((d_sorted_positions[d_cell_first_particles_indices[cellid]].x) * one_over_dx + 0.5f) - 1;
    smallest_nodes[cellid].y = (int)((d_sorted_positions[d_cell_first_particles_indices[cellid]].y) * one_over_dx + 0.5f) - 1;
    smallest_nodes[cellid].z = (int)((d_sorted_positions[d_cell_first_particles_indices[cellid]].z) * one_over_dx + 0.5f) - 1;
}

__global__ void undateGrid_kernal(const T dt, T** d_channels) {
    int idx = blockIdx.x;
    
    int cellid = (threadIdx.x);
    T tag = *((T*)((unsigned long long)d_channels[9] + idx * MEMOFFSET) + cellid);
    //printf("tag£»  %f\n", tag);
    if (tag > 0.000001f)
    {
		T mass = *((T*)((unsigned long long)d_channels[0] + idx * MEMOFFSET) + cellid);
        T C = *((T*)((unsigned long long)d_channels[7] + idx * MEMOFFSET) + cellid);
        //printf("C:  %f\n", C);
        T W = *((T*)((unsigned long long)d_channels[8] + idx * MEMOFFSET) + cellid);
        //*((T*)((unsigned long long)d_channels[8] + idx * MEMOFFSET) + cellid) = 0;
		//if (abs(W) < (T)1e-20) {
		//	//mass = 0;
		//	*((T*)((unsigned long long)d_channels[7] + idx * MEMOFFSET) + cellid) = 0.f;
		//	*((T*)((unsigned long long)d_channels[9] + idx * MEMOFFSET) + cellid) = 0.f;
		//}
		//else {
		//	*((T*)((unsigned long long)d_channels[7] + idx * MEMOFFSET) + cellid) = C / W;
		//	//mass = 1.f / mass;
		//}
/*		if (abs(mass) < (T)1e-32) {
            *((T*)((unsigned long long)d_channels[7] + idx * MEMOFFSET) + cellid) = 0.f;
            *((T*)((unsigned long long)d_channels[9] + idx * MEMOFFSET) + cellid) = 0.f;
			mass = 0;
		}
		else */{
			mass = 1.f / mass;
            *((T*)((unsigned long long)d_channels[7] + idx * MEMOFFSET) + cellid) = __fdividef(C, W);
		}
		//mass = 1.f / mass;
        //for (int i = 0; i < 3; ++i)
        *((T*)((unsigned long long)d_channels[1] + idx * MEMOFFSET) + cellid) *= mass;
        *((T*)((unsigned long long)d_channels[2] + idx * MEMOFFSET) + cellid) *= mass;
        *((T*)((unsigned long long)d_channels[3] + idx * MEMOFFSET) + cellid) *= mass;
        //if (*((T*)((unsigned long long)d_channels[0] + idx * MEMOFFSET) + cellid) != 0)
        *((T*)((unsigned long long)d_channels[2] + idx * MEMOFFSET) + cellid) +=  -10 * dt;
        mass = dt * mass;
        for (int i = 0; i < Dim; i++) {
            T test = *((T*)((unsigned long long)d_channels[i + 4] + idx * MEMOFFSET) + cellid) * mass;

            if (!isnan(test) && !isinf(test)) {
                *((T*)((unsigned long long)d_channels[i + 1] + idx * MEMOFFSET) + cellid) += test;
            }
        }

    }
}





__global__
void collideWithGround(unsigned long long* masks,
    unsigned long long* pageOffsets,
    T** d_channels)
{
	int idx = blockIdx.x;
	int cell = (threadIdx.x);

    int ci = cell / 16;
    int ck = cell - ci * 16;
    int cj = ck / 4;
    ck = ck % 4;

    int i = Bit_Pack_Mine(masks[0], pageOffsets[idx]) + ci;
    int j = Bit_Pack_Mine(masks[1], pageOffsets[idx]) + cj;
    int k = Bit_Pack_Mine(masks[2], pageOffsets[idx]) + ck;

    // sticky
    if (i <= 4) if (*((T*)((unsigned long long)d_channels[1] + idx * MEMOFFSET) + cell) < 0.f) {
        *((T*)((unsigned long long)d_channels[1] + idx * MEMOFFSET) + cell) = 0.f;
        *((T*)((unsigned long long)d_channels[2] + idx * MEMOFFSET) + cell) = 0.f;
        *((T*)((unsigned long long)d_channels[3] + idx * MEMOFFSET) + cell) = 0.f;
    }
    if (i > N - 4) if (*((T*)((unsigned long long)d_channels[1] + idx * MEMOFFSET) + cell) > 0.f) {
        *((T*)((unsigned long long)d_channels[1] + idx * MEMOFFSET) + cell) = 0.f;
        *((T*)((unsigned long long)d_channels[2] + idx * MEMOFFSET) + cell) = 0.f;
        *((T*)((unsigned long long)d_channels[3] + idx * MEMOFFSET) + cell) = 0.f;
    }
    if (j <= 4) if (*((T*)((unsigned long long)d_channels[2] + idx * MEMOFFSET) + cell) < 0.f) {
        *((T*)((unsigned long long)d_channels[2] + idx * MEMOFFSET) + cell) = 0.f;
        *((T*)((unsigned long long)d_channels[1] + idx * MEMOFFSET) + cell) = 0.f;
        *((T*)((unsigned long long)d_channels[3] + idx * MEMOFFSET) + cell) = 0.f;
    }
    if (j > N - 4) if (*((T*)((unsigned long long)d_channels[2] + idx * MEMOFFSET) + cell) > 0.f) {
        *((T*)((unsigned long long)d_channels[2] + idx * MEMOFFSET) + cell) = 0.f;
        *((T*)((unsigned long long)d_channels[1] + idx * MEMOFFSET) + cell) = 0.f;
        *((T*)((unsigned long long)d_channels[3] + idx * MEMOFFSET) + cell) = 0.f;
    }
    if (k <= 4) if (*((T*)((unsigned long long)d_channels[3] + idx * MEMOFFSET) + cell) < 0.f) {
        *((T*)((unsigned long long)d_channels[3] + idx * MEMOFFSET) + cell) = 0.f;
        *((T*)((unsigned long long)d_channels[1] + idx * MEMOFFSET) + cell) = 0.f;
        *((T*)((unsigned long long)d_channels[2] + idx * MEMOFFSET) + cell) = 0.f;
    }
    if (k > N - 4) if (*((T*)((unsigned long long)d_channels[3] + idx * MEMOFFSET) + cell) > 0.f) {
        *((T*)((unsigned long long)d_channels[3] + idx * MEMOFFSET) + cell) = 0.f;
        *((T*)((unsigned long long)d_channels[1] + idx * MEMOFFSET) + cell) = 0.f;
        *((T*)((unsigned long long)d_channels[2] + idx * MEMOFFSET) + cell) = 0.f;
    }
}



__global__
void PullGrid_kernal(unsigned long long* masks,
    unsigned long long* pageOffsets,
    T** d_channels, T t, T dx)
{
    
    int idx = blockIdx.x;
    int cell = (threadIdx.x);

    vector3T left, right;
    left.y = 0.3f;
    left.z = 0.25f;
    left.x = 0.5f;

    right.y = 0.3f;
    right.z = 0.75f;
    right.x = 0.5f;
    
    T L = 0.2f;
    //if(L<)

    T vv = 0.2f;

    if (right.z < 0.9f) {
        right.z += t * vv;
        left.z -= t * vv;
    }else{
        vv = 0.f;
    }


    int ci = cell / 16;
    int ck = cell - ci * 16;
    int cj = ck / 4;
    ck = ck % 4;

    int i = Bit_Pack_Mine(masks[0], pageOffsets[idx]) + ci;
    int j = Bit_Pack_Mine(masks[1], pageOffsets[idx]) + cj;
    int k = Bit_Pack_Mine(masks[2], pageOffsets[idx]) + ck;

    T nodeX = i * dx;
    T nodeY = j * dx;
    T nodeZ = k * dx;

    T disL = (nodeX - left.x) * (nodeX - left.x) + (nodeY - left.y) * (nodeY - left.y) + (nodeZ - left.z) * (nodeZ - left.z);
    T disR = (nodeX - right.x) * (nodeX - right.x) + (nodeY - right.y) * (nodeY - right.y) + (nodeZ - right.z) * (nodeZ - right.z);
    T range = L * L;
    if (disL < range) {
        *((T*)((unsigned long long)d_channels[1] + idx * MEMOFFSET) + cell) = 0.f;
        *((T*)((unsigned long long)d_channels[2] + idx * MEMOFFSET) + cell) = 0.f;
        *((T*)((unsigned long long)d_channels[3] + idx * MEMOFFSET) + cell) = -vv;
    }
    if (disR < range) {
        *((T*)((unsigned long long)d_channels[1] + idx * MEMOFFSET) + cell) = 0.f;
        *((T*)((unsigned long long)d_channels[2] + idx * MEMOFFSET) + cell) = 0.f;
        *((T*)((unsigned long long)d_channels[3] + idx * MEMOFFSET) + cell) = vv;
    }

    if (i <= 4) if (*((T*)((unsigned long long)d_channels[1] + idx * MEMOFFSET) + cell) < 0.f) {
        *((T*)((unsigned long long)d_channels[1] + idx * MEMOFFSET) + cell) = 0.f;
        *((T*)((unsigned long long)d_channels[2] + idx * MEMOFFSET) + cell) = 0.f;
        *((T*)((unsigned long long)d_channels[3] + idx * MEMOFFSET) + cell) = 0.f;
    }
    if (i > N - 4) if (*((T*)((unsigned long long)d_channels[1] + idx * MEMOFFSET) + cell) > 0.f) {
        *((T*)((unsigned long long)d_channels[1] + idx * MEMOFFSET) + cell) = 0.f;
        *((T*)((unsigned long long)d_channels[2] + idx * MEMOFFSET) + cell) = 0.f;
        *((T*)((unsigned long long)d_channels[3] + idx * MEMOFFSET) + cell) = 0.f;
    }
	T* vY = ((T*)((unsigned long long)d_channels[2] + idx * MEMOFFSET) + cell);
    if (j <= 4) if (*vY < 0.f) {
        *((T*)((unsigned long long)d_channels[2] + idx * MEMOFFSET) + cell) = 0.f;
        *((T*)((unsigned long long)d_channels[1] + idx * MEMOFFSET) + cell) = 0.f;
        *((T*)((unsigned long long)d_channels[3] + idx * MEMOFFSET) + cell) = 0.f;
		//*vY = (*vY) * -0.2f;
    }
    if (j > N - 4) if (*vY > 0.f) {
        *((T*)((unsigned long long)d_channels[2] + idx * MEMOFFSET) + cell) = 0.f;
        *((T*)((unsigned long long)d_channels[1] + idx * MEMOFFSET) + cell) = 0.f;
        *((T*)((unsigned long long)d_channels[3] + idx * MEMOFFSET) + cell) = 0.f;
		//*vY = (*vY) * -0.2f;
    }
    if (k <= 4) if (*((T*)((unsigned long long)d_channels[3] + idx * MEMOFFSET) + cell) < 0.f) {
        *((T*)((unsigned long long)d_channels[3] + idx * MEMOFFSET) + cell) = 0.f;
        *((T*)((unsigned long long)d_channels[1] + idx * MEMOFFSET) + cell) = 0.f;
        *((T*)((unsigned long long)d_channels[2] + idx * MEMOFFSET) + cell) = 0.f;
    }
    if (k > N - 4) if (*((T*)((unsigned long long)d_channels[3] + idx * MEMOFFSET) + cell) > 0.f) {
        *((T*)((unsigned long long)d_channels[3] + idx * MEMOFFSET) + cell) = 0.f;
        *((T*)((unsigned long long)d_channels[1] + idx * MEMOFFSET) + cell) = 0.f;
        *((T*)((unsigned long long)d_channels[2] + idx * MEMOFFSET) + cell) = 0.f;
    }

}

//__global__ void preG2P(T** d_channels) {
//    int idx = blockIdx.x;
//    int cellid = threadIdx.x;
//    for (int i = 0; i < Dim; i++) {
//        *((T*)((unsigned long long)d_channels[i + 7] + idx * MEMOFFSET) + cellid) = *((T*)((unsigned long long)d_channels[i + 1] + idx * MEMOFFSET) + cellid)
//            - *((T*)((unsigned long long)d_channels[i + 7] + idx * MEMOFFSET) + cellid);
//    }
//}




MPMTimeIntegrator::MPMTimeIntegrator(int transferScheme, int numParticle, T* dMemTrunk) :
    _transferScheme(transferScheme), _numParticle(numParticle), d_memTrunk(dMemTrunk)
{
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_contribution, sizeof(T) * numParticle * 9 + 2));
    //d_tmp = d_memTrunk;
}

MPMTimeIntegrator::~MPMTimeIntegrator() {
    CUDA_SAFE_CALL(cudaFree(d_contribution));
}

void MPMTimeIntegrator::initMatrix(unsigned int num_particle, T* matrix) {
    const unsigned int numthread = 256;
    const unsigned int numblock = (num_particle + numthread - 1) / numthread;
    initMatrix_kernal << <numblock, numthread >> > (num_particle, matrix);
}

void MPMTimeIntegrator::computeCellIndex(std::unique_ptr<DomainTransformer>& trans, std::unique_ptr<Particles>& particle) {
    const unsigned int numthread = 256;
    const unsigned int numblock = (trans->_numCell + numthread - 1) / numthread;
    calcIndex << <numblock, numthread >> > (trans->_numCell, one_over_dx, (trans->d_cell2particle),
        particle->d_orderedPos, particle->d_smallestNodeIndex);
}

void MPMTimeIntegrator::transferP2G(
    const T dt,
    std::unique_ptr<Particles>& geometry,
    std::unique_ptr<SPGrid>& grid,
    std::unique_ptr<DomainTransformer>& trans)
{
    grid->clear();
    const unsigned int blockNum = (unsigned int)trans->_numVirtualPage;
    const unsigned int threadNum = 512;

    //#if TRANSFER_SCHEME == 0
    //    P2G_FLIP << <blockNum, threadNum >> > (geometry->_numParticle, (const int*)trans->d_targetPage, (const int*)trans->d_virtualPageOffset,
    //        geometry->d_smallestNodeIndex,
    //        (const T*)d_contribution, trans->d_page2particle, trans->d_particle2cell, geometry->d_indices, geometry->d_indexTrans, geometry->d_orderedPos,
    //        geometry->d_orderedMass, geometry->d_orderedVel, grid->d_channels, trans->d_adjPage);
    //
    //    /*recordLaunch(std::string("P2G_FLIP"), (int)trans->_numVirtualPage, 512, (size_t)0, P2G_FLIP, geometry->_numParticle, (const int*)trans->d_targetPage, (const int*)trans->d_virtualPageOffset,
    //        (const int**)geometry->d_smallestNodeIndex,
    //        (const T*)d_contribution, trans->d_page2particle, trans->d_particle2cell, geometry->d_indices, geometry->d_indexTrans, geometry->d_orderedPos,
    //        geometry->d_orderedMass, geometry->d_orderedVel, grid->d_channels, trans->d_adjPage);*/
    //#endif
#if TRANSFER_SCHEME == 0

    P2G_APIC_CONFLICT_FREE << <blockNum, threadNum >> > (geometry->_numParticle, (const int*)trans->d_targetPage, (const int*)trans->d_virtualPageOffset,
        geometry->d_smallestNodeIndex, (const T*)d_contribution,
        trans->d_page2particle, trans->d_particle2cell, geometry->d_indices, geometry->d_indexTrans, geometry->d_orderedPos, geometry->d_orderedMass,
        geometry->d_orderedVel, geometry->d_B, grid->d_channels, trans->d_adjPage);

#endif
#if TRANSFER_SCHEME == 1

    P2G_APIC << <blockNum, threadNum >> > (geometry->_numParticle, 
        (const int*)trans->d_targetPage, 
        (const int*)trans->d_virtualPageOffset,
        geometry->d_smallestNodeIndex, 
        (const T*)d_contribution,
        trans->d_page2particle, 
        trans->d_particle2cell, 
        geometry->d_indices, 
        geometry->d_indexTrans, 
        geometry->d_orderedPos, 
        geometry->d_phase_C, 
        geometry->d_orderedMass,
        geometry->d_orderedVel, 
        geometry->d_B, 
        grid->d_channels, 
        trans->d_adjPage);

#endif
#if TRANSFER_SCHEME == 2
    P2G_MLS << <blockNum, threadNum >> > (geometry->_numParticle, 
		(const int*)trans->d_targetPage, 
		(const int*)trans->d_virtualPageOffset,
        geometry->d_smallestNodeIndex, 
		(const T*)d_contribution,
        trans->d_page2particle, 
		trans->d_particle2cell, 
		geometry->d_indices, 
		geometry->d_indexTrans, 
		geometry->d_orderedPos, 
		geometry->d_phase_C,
		geometry->d_orderedMass,
        geometry->d_orderedVel, 
		geometry->d_B, 
		dt, 
		grid->d_channels, 
		trans->d_adjPage, 
		trans->d_pageOffset);
    /*recordLaunch(std::string("P2G_MLS"), (int)trans->_numVirtualPage, 512, (size_t)0, P2G_MLS, geometry->_numParticle, (const int*)trans->d_targetPage, (const int*)trans->d_virtualPageOffset,
        (const int**)geometry->d_smallestNodeIndex, (const T*)d_contribution,
        trans->d_page2particle, trans->d_particle2cell, geometry->d_indices, geometry->d_indexTrans, geometry->d_orderedPos, geometry->d_orderedMass,
        geometry->d_orderedVel, geometry->d_B, dt, grid->d_channels, trans->d_adjPage, trans->d_pageOffset);*/
#endif
}





void MPMTimeIntegrator::undateGrid(const T dt, std::unique_ptr<SPGrid>& grid, std::unique_ptr<DomainTransformer>& trans)
{
    /*unsigned int numThread = 256;
    unsigned int numBlock = (trans->_numTotalPage + 3) / 4;*/
    undateGrid_kernal << < trans->_numTotalPage, 64 >> > (dt, grid->d_channels);
    //recordLaunch(std::string("ApplyGravity"), (trans->_numTotalPage), 64, (size_t)0, applyGravity, dt, trans->_numTotalPage, grid->d_channels);
}





void MPMTimeIntegrator::PullGrid(std::unique_ptr<SPGrid>& grid, std::unique_ptr<DomainTransformer>& trans, T dt)
{

    PullGrid_kernal << <trans->_numTotalPage, 64 >> > (grid->d_masks, trans->d_pageOffset, grid->d_channels, dt, dx);
}




void MPMTimeIntegrator::resolveCollision(std::unique_ptr<SPGrid>& grid, std::unique_ptr<DomainTransformer>& trans)
{
	
    collideWithGround << <trans->_numTotalPage, 64 >> > (grid->d_masks, trans->d_pageOffset, grid->d_channels);
}

//void MPMTimeIntegrator::call_preG2P(std::unique_ptr<SPGrid>& grid, std::unique_ptr<DomainTransformer>& trans)
//{
//    if (_transferScheme == 0)
//        preG2P << <(int)trans->_numTotalPage, 64 >> > (grid->d_channels);
//}

void MPMTimeIntegrator::transferG2P(const T dt, std::unique_ptr<Particles>& geometry, std::unique_ptr<SPGrid>& grid,
    std::unique_ptr<DomainTransformer>& trans)
{
    const unsigned int blockNum = (unsigned int)trans->_numVirtualPage;
    const unsigned int threadNum = 512;

    //#if TRANSFER_SCHEME == 0
    //    /*vector3T* vel;
    //    cudaMalloc((void**)&vel, sizeof(vector3T) * geometry->_numParticle);*/
    //
    //    G2P_FLIP << <blockNum, threadNum >> > (geometry->_numParticle, (const int*)trans->d_targetPage,
    //        (const int*)trans->d_virtualPageOffset,
    //        geometry->d_smallestNodeIndex,
    //        trans->d_page2particle, trans->d_particle2cell, geometry->d_indices, geometry->d_indexTrans, geometry->d_orderedPos,
    //        geometry->d_orderedVel, grid->d_channels, geometry->d_F, d_memTrunk, dt, trans->d_adjPage);
    //    /*CUDA_SAFE_CALL(cudaMemcpy(geometry->d_orderedVel, vel, sizeof(vector3T) * geometry->_numParticle, cudaMemcpyDeviceToDevice));
    //    cudaFree(vel);*/
    //    CUDA_SAFE_CALL(cudaMemcpy(geometry->d_orderedVel, d_memTrunk, sizeof(vector3T) * geometry->_numParticle, cudaMemcpyDeviceToDevice));
    //    CUDA_SAFE_CALL(cudaMemcpy(geometry->d_F, d_memTrunk + geometry->_numParticle * 3, sizeof(T) * geometry->_numParticle * 9, cudaMemcpyDeviceToDevice));
    //#endif      
#if TRANSFER_SCHEME == 0
    G2P_APIC_CONFLICT_FREE << <blockNum, threadNum >> > (geometry->_numParticle, (const int*)trans->d_targetPage, (const int*)trans->d_virtualPageOffset,
        geometry->d_smallestNodeIndex,
        trans->d_page2particle, trans->d_particle2cell, geometry->d_indices,
        geometry->d_indexTrans, geometry->d_orderedPos, geometry->d_orderedVel, grid->d_channels, geometry->d_F, geometry->d_B,
        d_memTrunk, dt, trans->d_adjPage);

    CUDA_SAFE_CALL(cudaMemcpy(geometry->d_F, d_memTrunk + geometry->_numParticle * 3, sizeof(T) * geometry->_numParticle * 9, cudaMemcpyDeviceToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(geometry->d_B, d_memTrunk + geometry->_numParticle * 12, sizeof(T) * geometry->_numParticle * 9, cudaMemcpyDeviceToDevice));
#endif  
#if TRANSFER_SCHEME == 1
    G2P_APIC << <blockNum, threadNum >> > (geometry->_numParticle, 
        (const int*)trans->d_targetPage, 
        (const int*)trans->d_virtualPageOffset,
        geometry->d_smallestNodeIndex,
        trans->d_page2particle, 
        trans->d_particle2cell, 
        geometry->d_indices,
        geometry->d_indexTrans, 
        geometry->d_orderedPos, 
        geometry->d_phase_C, 
        geometry->d_phase_C_sort, 
        geometry->d_orderedVel, 
        grid->d_channels, 
        geometry->d_F, 
        geometry->d_B,
        d_memTrunk, dt, 
        trans->d_adjPage);
    CUDA_SAFE_CALL(cudaMemcpy(geometry->d_phase_C, geometry->d_phase_C_sort, sizeof(T) * geometry->_numParticle, cudaMemcpyDeviceToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(geometry->d_F, d_memTrunk, sizeof(T) * geometry->_numParticle * 9, cudaMemcpyDeviceToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(geometry->d_B, d_memTrunk + geometry->_numParticle * 9, sizeof(T) * geometry->_numParticle * 9, cudaMemcpyDeviceToDevice));
#endif
#if TRANSFER_SCHEME == 2
    G2P_MLS << <blockNum, threadNum >> > (geometry->_numParticle, (const int*)trans->d_targetPage,
        (const int*)trans->d_virtualPageOffset,
        geometry->d_smallestNodeIndex,
        trans->d_page2particle, trans->d_particle2cell, geometry->d_indices, geometry->d_indexTrans, geometry->d_orderedPos, geometry->d_phase_C,
		geometry->d_phase_C_sort,
        geometry->d_orderedVel, grid->d_channels, geometry->d_F, geometry->d_B, d_memTrunk, dt, trans->d_adjPage);
	CUDA_SAFE_CALL(cudaMemcpy(geometry->d_phase_C, geometry->d_phase_C_sort, sizeof(T) * geometry->_numParticle, cudaMemcpyDeviceToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(geometry->d_B, d_memTrunk, sizeof(T) * geometry->_numParticle * 9, cudaMemcpyDeviceToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(geometry->d_F, d_memTrunk + geometry->_numParticle * 9, sizeof(T) * geometry->_numParticle * 9, cudaMemcpyDeviceToDevice));
#endif
}