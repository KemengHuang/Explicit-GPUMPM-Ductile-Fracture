#include "Particles.cuh"
#include"cuda_tools.h"
//#include<fstream>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include"cuda_kernal_tools.cuh"
//#define LOG_NUM_BANKS	 5
//#define CONFLICT_FREE_OFFSET(index) ((index) >> LOG_NUM_BANKS)


/*******************************************************************************************************/
//////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////Kernal_Function_Device////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////
/*******************************************************************************************************/

//__device__ unsigned long long Bit_Spread_Mine(const unsigned long long mask, const int data) {
//    unsigned long long rmask = __brevll(mask);
//    int dat = data;
//    unsigned long long result = 0;
//    unsigned char lz, offset = __clzll(rmask);
//    while (rmask) {
//        lz = __clzll(rmask) + 1;
//        result = result << lz | (dat & 1);
//        dat >>= 1, rmask <<= lz;
//    }
//    result = __brevll(result) >> __clzll(mask);
//    return result;
//}

__device__ unsigned long long Bit_Spread_Mine(const unsigned long long& mask, const int& data) {
    unsigned long long rmask = __brevll(mask);
    int dat = data;
    unsigned long long result = 0;
    unsigned char lz, offset = __clzll(rmask);
    unsigned int count = 0;
    while (rmask && dat) {
        lz = __clzll(rmask) + 1;
        result = result << lz | (dat & 1);
        dat >>= 1, rmask <<= lz;
        count += lz;
    }
    result = __brevll(result) >> (64 - count);//__clzll(mask);
    return result;
}
__device__ bool atomicMaxf(T* address, T val) {
    int* address_as_i = (int*)address;
    int old = *address_as_i, assumed;
    if (*address >= val) return false;
    do {
        assumed = old;
        old = atomicCAS(address_as_i, assumed,
            __float_as_int(fmaxf(val, __int_as_float(assumed))));
    } while (assumed != old);
    return true;
}

__global__ void calcMaxVel(const int numParticle, const vector3T* d_vel, T* _maxVelSquared) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= numParticle) return;

    T vel_squared = d_vel[idx].x * d_vel[idx].x;
    vel_squared += d_vel[idx].y * d_vel[idx].y;
    vel_squared += d_vel[idx].z * d_vel[idx].z;

    atomicMaxf(_maxVelSquared, vel_squared);
}

__global__ void calVelSqur(const int numParticle, const vector3T* d_vel, T* squr) {
	//int tid = threadIdx.x;
	int idof = blockIdx.x * blockDim.x;
	int idx = threadIdx.x + idof;
	extern __shared__ T tep[];

	if (idx >= numParticle) return;
    vector3T vel = d_vel[idx];
    T temp = vel.x * vel.x;
	temp += vel.y * vel.y;
	temp += vel.z * vel.z;

	int warpTid = threadIdx.x % 32;
	int warpId = (threadIdx.x >> 5);
	T nextTp;
	int warpNum;
	if (blockIdx.x == gridDim.x-1) {
		warpNum = ((numParticle - idof + 31) >> 5);
	}
	else {
		warpNum = ((blockDim.x) >> 5);
	}
	for (int i = 1; i < 32; i = (i << 1)) {
		nextTp = __shfl_down(temp, i);
		if (temp < nextTp) {
			temp = nextTp;
		}
	}
	if (warpTid == 0) {
		tep[warpId] = temp;
	}
	__syncthreads();
	if (threadIdx.x >= warpNum) return;
	if (warpNum > 1) {	
		temp = tep[threadIdx.x];
		for (int i = 1; i < warpNum; i = (i << 1)) {
			nextTp = __shfl_down(temp, i);
			if (temp < nextTp) {
				temp = nextTp;
			}
		}
		/*if (warpTid == 0) {
			tep[warpId] = temp;
		}*/
	}
	if (threadIdx.x == 0) {
		squr[blockIdx.x] = temp;
	}
}

__global__ void calVelSqur_b(const int numParticle, const vector3T* d_vel, T* squr) {
    //int tid = threadIdx.x;
    int ds = blockDim.x << 1;
    int idof = (blockIdx.x * ds);
    int idx = threadIdx.x + idof;
    extern __shared__ T tep[];

    if (idx >= numParticle) return;
    vector3T vel = d_vel[idx];
    T vel_squared = vel.x * vel.x;
    vel_squared += vel.y * vel.y;
    vel_squared += vel.z * vel.z;
    //int cfid = tid + CONFLICT_FREE_OFFSET(tid);
    tep[threadIdx.x] = vel_squared;

    int nextDb = idx + blockDim.x;
    if (nextDb < numParticle) {
        vel = d_vel[nextDb];
        vel_squared = vel.x * vel.x;
        vel_squared += vel.y * vel.y;
        vel_squared += vel.z * vel.z;
        tep[threadIdx.x + blockDim.x] = vel_squared;
    }

    __syncthreads();

    unsigned int bi = 0;
    for (int s = 1; s < ds; s = (s << 1)) {
        unsigned int kid = threadIdx.x << (bi + 1);
        if ((kid + s) >= ds || (idof + kid + s) >= numParticle) break;
        tep[kid] = tep[kid] > tep[kid + s] ? tep[kid] : tep[kid + s];
        ++bi;
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        squr[blockIdx.x] = tep[0];
    }
}

__global__ void cal_cellId(const int numParticle, const T one_over_dx, const unsigned long long* _masks, vector3T* _pos, unsigned long long* _offsets) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numParticle) return;

    _offsets[idx] |= Bit_Spread_Mine(_masks[0], (int)((_pos[idx].x) * one_over_dx + 0.5f) - 1);
    _offsets[idx] |= Bit_Spread_Mine(_masks[1], (int)((_pos[idx].y) * one_over_dx + 0.5f) - 1);
    _offsets[idx] |= Bit_Spread_Mine(_masks[2], (int)((_pos[idx].z) * one_over_dx + 0.5f) - 1);
}

__global__ void registerPage(const int numParticle, const int tableSize,
    const unsigned long long* d_cellId, unsigned long long* d_keyTable, int* d_valTable, int* d_numBucket) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= numParticle) return;
    unsigned long long p_pageId = d_cellId[idx] >> 12;
    unsigned long long hashkey = p_pageId % tableSize;
    unsigned long long ori;

    while ((ori = d_keyTable[hashkey]) != p_pageId) {
        if (ori == 0xffffffffffffffff)
            ori = atomicCAS((unsigned long long*)d_keyTable + hashkey, 0xffffffffffffffff, p_pageId);    ///< -1 is the default value, means unoccupied
        if (d_keyTable[hashkey] == p_pageId) {  ///< haven't found related record, so create a new entry
            if (ori == 0xffffffffffffffff) {
                d_valTable[hashkey] = atomicAdd(d_numBucket, 1);   ///< created a record
            }
            break;
        }
        hashkey += 127; ///< search next entry
        if (hashkey >= tableSize) hashkey = hashkey % tableSize;
    }
}

__global__ void registerPage_km(const int numParticle, 
	const unsigned long long* d_cellId, int* d_numBucket, int* pageId) {
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	if (idx >= numParticle) return;
	unsigned long long p_pageId = d_cellId[idx] >> 12;
	int ori;
	if ((ori = pageId[p_pageId]) == 0x00000000) {
		ori = atomicCAS((int*)pageId + p_pageId, 0x00000000, 1);
	}
	/*if (ori == 0x00000000) {
		atomicAdd(d_numBucket, 1);   ///< created a record
	}*/
}

__global__ void findPage_km(const int numParticle, 
    const unsigned long long* d_particleOffsets, int* d_particle2bucket, int* d_bucketSizes, int* d_page_offset) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= numParticle) return;
    unsigned long long key = d_particleOffsets[idx] >> 12;
    int bucketid = d_page_offset[key];
    int cellid = (d_particleOffsets[idx] & 0xfc) >> 2;
    bucketid = (bucketid << 6) | cellid;
    d_particle2bucket[idx] = bucketid;
    atomicAdd(d_bucketSizes + bucketid, 1);
}

__global__ void findPage(const int numParticle, const int tableSize,
	const unsigned long long* d_particleOffsets, unsigned long long* d_keyTable, int* d_valTable, int* d_particle2bucket, int* d_bucketSizes) {
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	if (idx >= numParticle) return;
	unsigned long long key = d_particleOffsets[idx] >> 12;

	unsigned long long hashkey = key % tableSize;
	while (d_keyTable[hashkey] != key) {
		hashkey += 127; ///< search next entry
		if (hashkey >= tableSize) hashkey = hashkey % tableSize;
	}
	int bucketid = d_valTable[hashkey];

	int cellid = (d_particleOffsets[idx] & 0xfc) >> 2;
	bucketid = (bucketid << 6) | cellid;
	d_particle2bucket[idx] = bucketid;
	atomicAdd(d_bucketSizes + bucketid, 1);
}

__global__ void reorderKey(const int numParticle, const int* d_particle2bucket, const int* d_offsets, int* d_sizes,
    const unsigned long long* d_keys, unsigned long long* d_orderedKeys, int* d_orderedIndices) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= numParticle) return;
    int bucketid = d_particle2bucket[idx];
    d_orderedKeys[bucketid = (d_offsets[bucketid] + atomicAdd(d_sizes + bucketid, 1))] = d_keys[idx];
    d_orderedIndices[bucketid] = idx;
}

__global__ void updateIndices(const int numParticle, int* d_indices_old, int* d_indices_map, int* d_indices_new)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numParticle) return;

    d_indices_new[idx] = d_indices_old[d_indices_map[idx]];
}

__global__ void gather3DShared(int numParticle, const int* _map, vector3T* _ori) {
    extern __shared__ T s_Vals[];
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numParticle) return;
    s_Vals[threadIdx.x] = _ori[_map[idx]].x;
    s_Vals[threadIdx.x + 512] = _ori[_map[idx]].y;
    s_Vals[threadIdx.x + 1024] = _ori[_map[idx]].z;
    __syncthreads();
    _ori[idx].x = s_Vals[threadIdx.x];
    _ori[idx].y = s_Vals[threadIdx.x + 512];
    _ori[idx].z = s_Vals[threadIdx.x + 1024];
}

__global__ void gather3D(int numParticle, const int* _map, const vector3T* _ori, vector3T* _ord) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numParticle) return;
    _ord[idx].x = _ori[_map[idx]].x;
    _ord[idx].y = _ori[_map[idx]].y;
    _ord[idx].z = _ori[_map[idx]].z;
}

__global__ void gather1D(int numParticle, const int* _map, const T* _ori, T* _ord) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numParticle) return;
    _ord[idx] = _ori[_map[idx]];

}

__global__ void updateColor(int numParticle, T* _ph, unsigned int* mcolor) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numParticle) return;
    T pha = _ph[idx];

    mcolor[idx] = (unsigned int((1) * 255) << 24) | (unsigned int((pha * 0.5f) * 255.0f) << 16) | (unsigned int(((1) * 0.5f) * 255.0f) << 8) | unsigned int(((1 - pha) * 0.5f) * 255.0f);

}

__global__
void mymaxF_b(T* mem, int numbers) {
    int tid = threadIdx.x;
    int ds = blockDim.x << 1;
    int idof = (blockIdx.x * ds);
    int idx = tid + idof;
    extern __shared__ T tep[];
    if (idx >= numbers) return;
    tep[tid] = mem[idx];
    int nextDb = idx + blockDim.x;
    if (nextDb < numbers) {
        tep[tid + blockDim.x] = mem[nextDb];
    }
    __syncthreads();
    unsigned int bi = 0;
    for (int s = 1; s < ds; s = (s << 1)) {
        unsigned int kid = tid << (bi + 1);
        if ((kid + s) >= ds || (idof + kid + s) >= numbers) break;
        tep[kid] = tep[kid] > tep[kid + s] ? tep[kid] : tep[kid + s];
        ++bi;
        __syncthreads();
    }
    if (tid == 0) {
        mem[blockIdx.x] = tep[0];
    }
}

__global__
void mymaxF(T* mem, int numbers) {
    //int tid = threadIdx.x;
    int idof = blockIdx.x * blockDim.x;
    int idx = threadIdx.x + idof;

    extern __shared__ T tep[];
    
    if (idx >= numbers) return;
	//int cfid = tid + CONFLICT_FREE_OFFSET(tid);
    T temp = mem[idx];
	int warpTid = threadIdx.x % 32;
	int warpId = (threadIdx.x >> 5);
	T nextTp;
	int warpNum;
	//int tidNum = 32;
	if (blockIdx.x == gridDim.x-1) {
		//tidNum = numbers - idof;
		warpNum = ((numbers - idof + 31) >> 5);
	}
	else {
		warpNum = ((blockDim.x) >> 5);
	}
	for (int i = 1; i < 32; i = (i << 1)) {
		nextTp = __shfl_down(temp, i);
		if (temp < nextTp) {
			temp = nextTp;
		}
	}
	if (warpTid == 0) {
		tep[warpId] = temp;
	}
	__syncthreads();
	if (threadIdx.x >= warpNum) return;
	if (warpNum > 1) {
	//	tidNum = warpNum;
	    temp = tep[threadIdx.x];
	//	warpNum = ((tidNum + 31) >> 5);
		for (int i = 1; i < warpNum; i = (i << 1)) {
			nextTp = __shfl_down(temp, i);
			if (temp < nextTp) {
				temp = nextTp;
			}
		}
	}
	if (threadIdx.x == 0) {
		mem[blockIdx.x] = temp;
	}
}

__global__
void initArray(T* d_array, int numbers, T val) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= numbers) return;
    d_array[idx] = val;
}

__global__
void initIndex(int* index, int numbers) {
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx >= numbers) return;
	index[idx] = idx;
}


/********************************************************************************************************/
////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////HOST_FUNCTION//////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////
/*******************************************************************************************************/

void Particles::getMaxParticleVel_b(T* maxVelSqared) {

    int numP = _numParticle;
    const unsigned int threadNum = 256;
    int dataSize = threadNum * 2;
    int blockNum = (numP + dataSize - 1) / dataSize;
    //int memoff = (threadNum+(1 << LOG_NUM_BANKS)-1)>> LOG_NUM_BANKS;
    unsigned int sharedMsize = sizeof(T) * (dataSize);
    calcMaxVel << <blockNum, threadNum >> > (_numParticle, d_orderedVel, maxVelSqared);

    calVelSqur_b << <blockNum, threadNum, sharedMsize >> > (_numParticle, d_orderedVel, velSqu);
    numP = blockNum;
    blockNum = (numP + dataSize - 1) / dataSize;

    while (numP > 1) {
        mymaxF_b << <blockNum, threadNum, sharedMsize >> > (velSqu, numP);
        numP = blockNum;
        blockNum = (numP + dataSize - 1) / dataSize;

    }
    cudaMemcpy(maxVelSqared, velSqu, sizeof(T), cudaMemcpyDeviceToHost);
}

void Particles::getMaxParticleVel(T* maxVelSqared) {

    int numP = _numParticle;
    const unsigned int threadNum = 256;
    int blockNum = (numP + threadNum - 1) / threadNum;

    calcMaxVel << <blockNum, threadNum >> > (_numParticle, d_orderedVel, maxVelSqared);
}

void Particles::getMaxParticleVel_c(T* maxVelSqared) {

    int numP = _numParticle;
    const unsigned int threadNum = 128;
    int blockNum = (numP + threadNum - 1) / threadNum;
	
	unsigned int sharedMsize = sizeof(T) * (threadNum >> 5);
   
    calVelSqur << <blockNum, threadNum, sharedMsize >> > (_numParticle, d_orderedVel, velSqu);
	numP = blockNum;
	blockNum = (numP + threadNum - 1) / threadNum;

	while (numP > 1) {
        mymaxF << <blockNum, threadNum, sharedMsize >> > (velSqu, numP);
        numP = blockNum;
        blockNum = (numP + threadNum - 1) / threadNum;

    }
    cudaMemcpy(maxVelSqared, velSqu, sizeof(T), cudaMemcpyDeviceToHost);
}
void Particles::initialize_kernal(const T* h_mass, const vector3T* h_pos, vector3T* h_vel) {

    //CUDA_SAFE_CALL(cudaMemset(d_phase_C, 1.f, sizeof(unsigned long long) * _numParticle));
    CUDA_SAFE_CALL(cudaMemcpy(d_mass, h_mass, sizeof(T) * _numParticle, cudaMemcpyHostToDevice));	// or use thrust::copy
    CUDA_SAFE_CALL(cudaMemcpy(d_pos, h_pos, sizeof(vector3T) * _numParticle, cudaMemcpyHostToDevice));	// or use thrust::copy

    // calculate offsets
    const unsigned int threadNum = 256;
    const unsigned int blockNum = (_numParticle + threadNum - 1) / threadNum;

    initArray << <blockNum, threadNum >> > (d_maxPsi, _numParticle, 0.f);
    initArray << <blockNum, threadNum >> > (d_phase_C, _numParticle, 1.f);
    initArray << <blockNum, threadNum >> > (d_g, _numParticle, 1.f);
    initArray << <blockNum, threadNum >> > (d_alpha, _numParticle, 0.f);

    CUDA_SAFE_CALL(cudaMemset(cell_id, 0, sizeof(unsigned long long) * _numParticle));
    cal_cellId << <blockNum, threadNum >> > (_numParticle, one_over_dx, _dmasks, d_pos, cell_id);
    //thrust::sequence(thrust::device_ptr<int>(d_indices), thrust::device_ptr<int>(d_indices) + _numParticle);
	initIndex << <blockNum, threadNum >> > (d_indices, _numParticle);

    sort_by_offsets(blockNum, threadNum);
	printf("during particle ordering    num particle: %d, num bucket: %d(%d)\n", _numParticle, _numBucket, _numBucket >> 6);

    gather3D << <blockNum, threadNum >> > (_numParticle, d_indices, d_pos, d_orderedPos);


    CUDA_SAFE_CALL(cudaMemcpy(d_orderedVel, h_vel, sizeof(vector3T) * _numParticle, cudaMemcpyHostToDevice));

    //CUDA_SAFE_CALL(cudaMemcpy(h_vel, d_orderedVel, sizeof(vector3T) * _numParticle, cudaMemcpyDeviceToHost));

    /*vector3T* hv = new vector3T[_numParticle];
    cudaMemcpy(hv, d_orderedPos, sizeof(vector3T)* _numParticle, cudaMemcpyDeviceToHost);
    vector3T aa = hv[0];
    std::ofstream outt("vel_test.txt");
    for(int i=0;i<_numParticle;i++)
        outt << hv[i].x << "   " << hv[i].y << "   " << hv[i].z << std::endl;
    outt.close();*/

    //gather3DShared << <(_numParticle + 511) / 512, 512, sizeof(T) * 512 * 3 >> > (_numParticle, d_indices, d_orderedVel);
    CUDA_SAFE_CALL(cudaMemcpy(d_orderedMass, d_mass, sizeof(T) * _numParticle, cudaMemcpyDeviceToDevice));
}


void Particles::sort_by_offsets(const unsigned int& blockNum, const unsigned int threadNum) {
    //////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////SORT_OFFSET////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////
    CUDA_SAFE_CALL(cudaMemcpy((unsigned long long*)d_memTrunk + _numParticle, d_indices, sizeof(int) * _numParticle, cudaMemcpyDeviceToDevice));
    /// histogram sort (substitution of radix sort)
    CUDA_SAFE_CALL(cudaMemset(d_numBucket, 0, sizeof(int)));
    CUDA_SAFE_CALL(cudaMemset(d_keyTable, 0xffffffffffffffff, sizeof(unsigned long long) * _tableSize));
    // to write for sorting optimization

    

	registerPage << <blockNum, threadNum >> > (_numParticle, _tableSize, cell_id, d_keyTable, d_valTable, d_numBucket);
	CUDA_SAFE_CALL(cudaMemcpy((void**)&_numBucket, d_numBucket, sizeof(int), cudaMemcpyDeviceToHost));

	//CUDA_SAFE_CALL(cudaMemset(d_pageId, 0x00000000, sizeof(int) * space_page_num));
    //registerPage_km << <blockNum, threadNum >> > (_numParticle, cell_id, d_numBucket, d_pageId);
	//if (space_page_num < pretype_threshold) {
	//	Ex_Prefix_Sum_Int(d_pageId, d_page_offset, space_page_num);
	//}
	//else {
	//	thrust::exclusive_scan(thrust::device_ptr<int>(d_pageId), thrust::device_ptr<int>(d_pageId) + space_page_num, thrust::device_ptr<int>(d_page_offset));
	//}
	//CUDA_SAFE_CALL(cudaMemcpy((void**)&_numBucket, d_page_offset + space_page_num - 1, sizeof(int), cudaMemcpyDeviceToHost));

	
	
	_numBucket <<= 6;
	//printf("during particle ordering    num particle: %d, num bucket: %d(%d)\n", _numParticle, _numBucket, _numBucket >> 6);
	CUDA_SAFE_CALL(cudaMemset(d_bucketSizes, 0, sizeof(int) * _numBucket));

	findPage << <blockNum, threadNum >> > (_numParticle, _tableSize, (const unsigned long long*)cell_id, d_keyTable, d_valTable, d_particle2bucket, d_bucketSizes);
	//findPage_km << <blockNum, threadNum >> > (_numParticle, (const unsigned long long*)cell_id, d_particle2bucket, d_bucketSizes, d_page_offset);
	
	if (_numBucket < pretype_threshold) {
		
		Ex_Prefix_Sum_Int(d_bucketSizes, d_bucketOffsets, _numBucket);
	}
	else {
		thrust::exclusive_scan(thrust::device_ptr<int>(d_bucketSizes), thrust::device_ptr<int>(d_bucketSizes) + _numBucket, thrust::device_ptr<int>(d_bucketOffsets));
	}
    CUDA_SAFE_CALL(cudaMemset(d_bucketSizes, 0, sizeof(int) * _numBucket));


    CUDA_SAFE_CALL(cudaMemcpy(d_memTrunk, cell_id, sizeof(unsigned long long) * _numParticle, cudaMemcpyDeviceToDevice));


    reorderKey << <blockNum, threadNum >> > (_numParticle, (const int*)d_particle2bucket, (const int*)d_bucketOffsets, d_bucketSizes,
        (const unsigned long long*)d_memTrunk, cell_id, d_indices);


    // d_indexTrans from last timestep to current
    CUDA_SAFE_CALL(cudaMemcpy(d_indexTrans, d_indices, sizeof(int) * _numParticle, cudaMemcpyDeviceToDevice));
    updateIndices << <blockNum, threadNum >> > (_numParticle, (int*)d_memTrunk + 2 * _numParticle, d_indexTrans, d_indices);

    ////////////////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////END_SORT///////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////////////////
}

void Particles::reorder() {
    const unsigned int threadNum = 512;
    const unsigned int blockNum = (_numParticle + threadNum - 1) / threadNum;

    CUDA_SAFE_CALL(cudaMemcpy(d_memTrunk, d_maxPsi, sizeof(T) * _numParticle, cudaMemcpyDeviceToDevice));
    gather1D << <blockNum, threadNum >> > (_numParticle, (const int*)d_indexTrans, d_memTrunk, d_maxPsi);

    CUDA_SAFE_CALL(cudaMemcpy(d_memTrunk, d_g, sizeof(T) * _numParticle, cudaMemcpyDeviceToDevice));
    gather1D << <blockNum, threadNum >> > (_numParticle, (const int*)d_indexTrans, d_memTrunk, d_g);

    CUDA_SAFE_CALL(cudaMemcpy(d_memTrunk, d_alpha, sizeof(T) * _numParticle, cudaMemcpyDeviceToDevice));
    gather1D << <blockNum, threadNum >> > (_numParticle, (const int*)d_indexTrans, d_memTrunk, d_alpha);

    CUDA_SAFE_CALL(cudaMemset(cell_id, 0, sizeof(unsigned long long) * _numParticle));



    updateColor << <blockNum, threadNum >> > (_numParticle, d_phase_C, d_color);

    cal_cellId << <blockNum, threadNum >> > (_numParticle, one_over_dx, _dmasks, d_orderedPos, cell_id);

    sort_by_offsets(blockNum, threadNum);

    // position
    CUDA_SAFE_CALL(cudaMemcpy(d_pos, d_orderedPos, sizeof(vector3T) * _numParticle, cudaMemcpyDeviceToDevice));

    gather3D << <blockNum, threadNum >> > (_numParticle, (const int*)d_indexTrans, d_pos, d_orderedPos);


}