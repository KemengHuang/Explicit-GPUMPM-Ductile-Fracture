#include "P2GKernal.cuh"
//#include "CudaDeviceUtils.cuh"
//#include <cstdio>
#define LOG_NUM_BANKS	 4
#define CONFLICT_FREE_OFFSET(index) ((index) >> LOG_NUM_BANKS)
__global__ void P2G_MLS(
    const int numParticle, const int* d_targetPages, const int* d_virtualPageOffsets, const int3* smallest_nodes,
    const T* d_sigma, int* d_block_offsets, int* d_cellids, int* d_indices, int* d_indexTrans, vector3T* d_sorted_positions, T* d_sorted_C, T* d_sorted_masses,
    vector3T* d_sorted_velocities, T* d_B, const T dt, T** d_channels, int** d_adjPage, unsigned long long* d_pageOffsets)
{
    __shared__ T buffer[7][8][8][8];
    int cellid = (7 * 8 * 8 * 8 + blockDim.x - 1) / blockDim.x;
    for (int i = 0; i < cellid; ++i)
        if (blockDim.x * i + threadIdx.x < 7 * 8 * 8 * 8)
            *((&buffer[0][0][0][0]) + blockDim.x * i + threadIdx.x) = (T)0;
    __syncthreads();

    int pageid = d_targetPages[blockIdx.x] - 1;
    cellid = d_block_offsets[pageid];
    int relParid = 512 * (blockIdx.x - d_virtualPageOffsets[pageid]) + threadIdx.x;
    int parid = cellid + relParid;

    int laneid = threadIdx.x & 0x1f;
    bool bBoundary;
    if (relParid < d_block_offsets[pageid + 1] - d_block_offsets[pageid])
    {
        cellid = d_cellids[parid] - 1;
        bBoundary = laneid == 0 || cellid + 1 != d_cellids[parid - 1];
    }
    else
        bBoundary = true;

    unsigned int mark = __ballot_sync(0xffffffff, bBoundary); // a bit-mask 
    mark = __brev(mark);
    unsigned int interval = min(__clz(mark << (laneid + 1)), 31 - laneid);
    mark = interval;
    for (int iter = 1; iter & 0x1f; iter <<= 1) {
        int tmp = __shfl_down_sync(__activemask(), mark, iter);
        mark = tmp > mark ? tmp : mark; /*if (tmp > mark) mark = tmp;*/
    }
    mark = __shfl_sync(0xffffffff, mark, 0);
    __syncthreads();

    int smallest_node[3];
    if (relParid < d_block_offsets[pageid + 1] - d_block_offsets[pageid]) {

        T wOneD[3][3];

        smallest_node[0] = smallest_nodes[cellid].x;
        smallest_node[1] = smallest_nodes[cellid].y;
        smallest_node[2] = smallest_nodes[cellid].z;

        int parid_trans = d_indexTrans[parid];
        T sig[9];
        sig[0] = d_sigma[parid_trans + (0) * numParticle]; sig[1] = d_sigma[parid_trans + (1) * numParticle]; sig[2] = d_sigma[parid_trans + (2) * numParticle];
        sig[3] = d_sigma[parid_trans + (3) * numParticle]; sig[4] = d_sigma[parid_trans + (4) * numParticle]; sig[5] = d_sigma[parid_trans + (5) * numParticle];
        sig[6] = d_sigma[parid_trans + (6) * numParticle]; sig[7] = d_sigma[parid_trans + (7) * numParticle]; sig[8] = d_sigma[parid_trans + (8) * numParticle];

        T B[9];
        B[0] = d_B[parid_trans + 0 * numParticle]; B[1] = d_B[parid_trans + 1 * numParticle]; B[2] = d_B[parid_trans + 2 * numParticle];
        B[3] = d_B[parid_trans + 3 * numParticle]; B[4] = d_B[parid_trans + 4 * numParticle]; B[5] = d_B[parid_trans + 5 * numParticle];
        B[6] = d_B[parid_trans + 6 * numParticle]; B[7] = d_B[parid_trans + 7 * numParticle]; B[8] = d_B[parid_trans + 8 * numParticle];

        T mass = d_sorted_masses[d_indices[parid]];

        for (int i = 0; i < 9; ++i)
            B[i] = (B[i] * mass - sig[i] * dt) * D_inverse;

        T xp[3];
        xp[0] = d_sorted_positions[parid].x - smallest_node[0] * dx;
        xp[1] = d_sorted_positions[parid].y - smallest_node[1] * dx;
        xp[2] = d_sorted_positions[parid].z - smallest_node[2] * dx;

        for (int v = 0; v < 3; ++v) {
            T d0 = xp[v] * one_over_dx;
            T z = ((T)1.5 - d0);
            wOneD[v][0] = (T)0.5 * z * z;
            d0 = d0 - 1.0f;
            wOneD[v][1] = (T)0.75 - d0 * d0;
            z = (T)1.5 - (1.0f - d0);
            wOneD[v][2] = (T)0.5 * z * z;
        }

        T vel[3];
        vel[0] = d_sorted_velocities[parid_trans].x;
        vel[1] = d_sorted_velocities[parid_trans].y;
        vel[2] = d_sorted_velocities[parid_trans].z;

        smallest_node[0] = smallest_node[0] & 0x3;
        smallest_node[1] = smallest_node[1] & 0x3;
        smallest_node[2] = smallest_node[2] & 0x3;

		T C = d_sorted_C[parid_trans];

        T val[7];
		T xi_minus_xp[3];
		T weight;
		T tmp[7];
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                for (int k = 0; k < 3; ++k) {
                    weight = wOneD[0][i] * wOneD[1][j] * wOneD[2][k];


                    val[0] = mass * weight;

                    
                    xi_minus_xp[0] = i * dx - xp[0];
                    xi_minus_xp[1] = j * dx - xp[1];
                    xi_minus_xp[2] = k * dx - xp[2];

                    val[1] = val[0] * vel[0];
                    val[2] = val[0] * vel[1];
                    val[3] = val[0] * vel[2];


                    val[1] += (B[0] * xi_minus_xp[0] + B[3] * xi_minus_xp[1] + B[6] * xi_minus_xp[2]) * weight;
                    val[2] += (B[1] * xi_minus_xp[0] + B[4] * xi_minus_xp[1] + B[7] * xi_minus_xp[2]) * weight;
                    val[3] += (B[2] * xi_minus_xp[0] + B[5] * xi_minus_xp[1] + B[8] * xi_minus_xp[2]) * weight;

					val[4] = weight * C;
					val[5] = weight;
					val[6] = 0.0001f;// wOneD[0][i] * wOneD[1][j] * wOneD[2][k];


                    for (int iter = 1; iter <= mark; iter <<= 1) {
                        tmp[7]; for (int i = 0; i < 7; ++i) tmp[i] = __shfl_down_sync(__activemask(), val[i], iter);
                        if (interval >= iter) for (int i = 0; i < 7; ++i) val[i] += tmp[i];
                    }

                    if (bBoundary) for (int ii = 0; ii < 7; ++ii)
                        atomicAdd(&(buffer[ii][smallest_node[0] + i][smallest_node[1] + j][smallest_node[2] + k]), val[ii]);

                }
            }
        }
    }
    __syncthreads();

    int block = threadIdx.x & 0x3f;
    int ci = block >> 4;
    int cj = (block & 0xc) >> 2;
    int ck = block & 3;

    block = threadIdx.x >> 6;
    int bi = block >> 2;
    int bj = (block & 2) >> 1;
    int bk = block & 1;

    int page_idx = block ? d_adjPage[block - 1][pageid] : pageid;

    for (int ii = 0; ii < 7; ++ii)
        if (buffer[ii][bi * 4 + ci][bj * 4 + cj][bk * 4 + ck] != 0)
            atomicAdd((T*)((unsigned long long)d_channels[ii+(ii/4)*3] + page_idx * 4096) + (ci * 16 + cj * 4 + ck), buffer[ii][bi * 4 + ci][bj * 4 + cj][bk * 4 + ck]);
}

__global__ void P2G_APIC_CONFLICT_FREE(
    const int numParticle,
    const int* d_targetPages,
    const int* d_virtualPageOffsets,
    const int3* smallest_nodes,
    const T* d_sigma,
    int* d_block_offsets,
    int* d_cellids,
    int* d_indices,
    int* d_indexTrans,
    vector3T* d_sorted_positions,
    T* d_sorted_masses,
    vector3T* d_sorted_velocities,
    T* d_B,
    T** d_channels,
    int** d_adjPage)
{
    __shared__ T buffer[1606]; // buffer[7][6][6][6];
    int kkk = 7 * 6 * 6 * 6;
    int cellid = (kkk + blockDim.x - 1) / blockDim.x;
    for (int i = 0; i < cellid; ++i) {
        int sdid = blockDim.x * i + threadIdx.x;
        if (sdid < kkk) {
            buffer[sdid + CONFLICT_FREE_OFFSET(sdid)] = 0.f;
        }
    }
    __syncthreads();

    int pageid = d_targetPages[blockIdx.x] - 1;
    cellid = d_block_offsets[pageid];
    int relParid = 512 * (blockIdx.x - d_virtualPageOffsets[pageid]) + threadIdx.x;
    int parid = cellid + relParid;

    int laneid = threadIdx.x & 0x1f;
    bool bBoundary;
    if (relParid < d_block_offsets[pageid + 1] - d_block_offsets[pageid])
    {
        cellid = d_cellids[parid] - 1;
        bBoundary = laneid == 0 || cellid + 1 != d_cellids[parid - 1];
    }
    else
        bBoundary = true;

    unsigned int mark = __ballot_sync(0xffffffff, bBoundary); // a bit-mask 
    mark = __brev(mark);
    unsigned int interval = min(__clz(mark << (laneid + 1)), 31 - laneid);
    mark = interval;
    for (int iter = 1; iter & 0x1f; iter <<= 1) {
        int tmp = __shfl_down_sync(__activemask(), mark, iter);
        mark = tmp > mark ? tmp : mark; /*if (tmp > mark) mark = tmp;*/
    }
    mark = __shfl_sync(0xffffffff, mark, 0);
    __syncthreads();

    int smallest_node[3];
    if (relParid < d_block_offsets[pageid + 1] - d_block_offsets[pageid]) {
        T wOneD[3][3], wgOneD[3][3];

        smallest_node[0] = smallest_nodes[cellid].x;
        smallest_node[1] = smallest_nodes[cellid].y;
        smallest_node[2] = smallest_nodes[cellid].z;

        T sig[9];
        int parid_trans = d_indexTrans[parid];
        sig[0] = d_sigma[parid_trans + (0) * numParticle]; sig[1] = d_sigma[parid_trans + (1) * numParticle]; sig[2] = d_sigma[parid_trans + (2) * numParticle];
        sig[3] = d_sigma[parid_trans + (3) * numParticle]; sig[4] = d_sigma[parid_trans + (4) * numParticle]; sig[5] = d_sigma[parid_trans + (5) * numParticle];
        sig[6] = d_sigma[parid_trans + (6) * numParticle]; sig[7] = d_sigma[parid_trans + (7) * numParticle]; sig[8] = d_sigma[parid_trans + (8) * numParticle];

        T B[9];
        B[0] = d_B[parid_trans + 0 * numParticle]; B[1] = d_B[parid_trans + 1 * numParticle]; B[2] = d_B[parid_trans + 2 * numParticle];
        B[3] = d_B[parid_trans + 3 * numParticle]; B[4] = d_B[parid_trans + 4 * numParticle]; B[5] = d_B[parid_trans + 5 * numParticle];
        B[6] = d_B[parid_trans + 6 * numParticle]; B[7] = d_B[parid_trans + 7 * numParticle]; B[8] = d_B[parid_trans + 8 * numParticle];
        for (int i = 0; i < 9; ++i)
            B[i] *= D_inverse;

        T xp[3];
        xp[0] = d_sorted_positions[parid].x - smallest_node[0] * dx;
        xp[1] = d_sorted_positions[parid].y - smallest_node[1] * dx;
        xp[2] = d_sorted_positions[parid].z - smallest_node[2] * dx;

        for (int v = 0; v < 3; ++v) {
            T d0 = xp[v] * one_over_dx;
            T z = ((T)1.5 - d0);
            wOneD[v][0] = (T)0.5 * z * z;
            wgOneD[v][0] = -z;
            d0 = d0 - 1;
            wOneD[v][1] = (T)0.75 - d0 * d0;
            wgOneD[v][1] = -d0 * 2;
            z = (T)1.5 - (1 - d0);
            wOneD[v][2] = (T)0.5 * z * z;
            wgOneD[v][2] = z;
        }

        wgOneD[0][0] *= one_over_dx;
        wgOneD[0][1] *= one_over_dx;
        wgOneD[0][2] *= one_over_dx;
        wgOneD[1][0] *= one_over_dx;
        wgOneD[1][1] *= one_over_dx;
        wgOneD[1][2] *= one_over_dx;
        wgOneD[2][0] *= one_over_dx;
        wgOneD[2][1] *= one_over_dx;
        wgOneD[2][2] *= one_over_dx;

        T vel[3];
        vel[0] = d_sorted_velocities[parid_trans].x;
        vel[1] = d_sorted_velocities[parid_trans].y;
        vel[2] = d_sorted_velocities[parid_trans].z;

        smallest_node[0] = smallest_node[0] & 0x3;
        smallest_node[1] = smallest_node[1] & 0x3;
        smallest_node[2] = smallest_node[2] & 0x3;

        T mass = d_sorted_masses[d_indices[parid]];

        T val[7];
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                for (int k = 0; k < 3; ++k) {

                    T wg[3];
                    wg[0] = wgOneD[0][i] * wOneD[1][j] * wOneD[2][k];
                    wg[1] = wOneD[0][i] * wgOneD[1][j] * wOneD[2][k];
                    wg[2] = wOneD[0][i] * wOneD[1][j] * wgOneD[2][k];

                    val[0] = mass * wOneD[0][i] * wOneD[1][j] * wOneD[2][k];
                    val[4] = -(sig[0] * wg[0] + sig[3] * wg[1] + sig[6] * wg[2]);
                    val[5] = -(sig[1] * wg[0] + sig[4] * wg[1] + sig[7] * wg[2]);
                    val[6] = -(sig[2] * wg[0] + sig[5] * wg[1] + sig[8] * wg[2]);

                    T xi_minus_xp[3];
                    xi_minus_xp[0] = i * dx - xp[0];
                    xi_minus_xp[1] = j * dx - xp[1];
                    xi_minus_xp[2] = k * dx - xp[2];

                    val[1] = vel[0];
                    val[2] = vel[1];
                    val[3] = vel[2];
                    val[1] += (B[0] * xi_minus_xp[0] + B[3] * xi_minus_xp[1] + B[6] * xi_minus_xp[2]);
                    val[2] += (B[1] * xi_minus_xp[0] + B[4] * xi_minus_xp[1] + B[7] * xi_minus_xp[2]);
                    val[3] += (B[2] * xi_minus_xp[0] + B[5] * xi_minus_xp[1] + B[8] * xi_minus_xp[2]);
                    val[1] *= val[0];
                    val[2] *= val[0];
                    val[3] *= val[0];

                    for (int iter = 1; iter <= mark; iter <<= 1) {
                        T tmp[7]; for (int i = 0; i < 7; ++i) tmp[i] = __shfl_down_sync(__activemask(), val[i], iter);
                        if (interval >= iter) for (int i = 0; i < 7; ++i) val[i] += tmp[i];
                    }

                    if (bBoundary) for (int ii = 0; ii < 7; ++ii) {
                        int sdid = ii * 216 + (smallest_node[0] + i) * 36 + (smallest_node[1] + j) * 6 + smallest_node[2] + k;
                        atomicAdd(buffer + sdid + CONFLICT_FREE_OFFSET(sdid), val[ii]);
                    }

                }
            }
        }
    }
    __syncthreads();

    int block = threadIdx.x & 0x3f;
    int ci = block >> 4;
    int cj = (block & 0xc) >> 2;
    int ck = block & 3;

    block = threadIdx.x >> 6;
    int bi = block >> 2;
    int bj = (block & 2) >> 1;
    int bk = block & 1;

    int page_idx = block ? d_adjPage[block - 1][pageid] : pageid;

    for (int ii = 0; ii < 7; ++ii)
    {
        int aa = bi * 4 + ci, bb = bj * 4 + cj, cc = bk * 4 + ck;
        if (aa < 6 && bb < 6 && cc < 6) {
            int sdid = ii * 216 + aa * 36 + bb * 6 + cc;
            T datatemp = buffer[sdid + CONFLICT_FREE_OFFSET(sdid)];
            if (datatemp != 0)
                atomicAdd((T*)((unsigned long long)d_channels[ii] + page_idx * 4096) + (ci * 16 + cj * 4 + ck), datatemp);
        }
    }
}

//__global__ void P2G_FLIP(  ///< use warp optimization
//    const int numParticle,
//    const int* d_targetPages,
//    const int* d_virtualPageOffsets,
//    const int3* smallest_nodes,
//    const T* d_sigma,
//    int* d_block_offsets,
//    int* d_cellids,
//    int* d_indices,
//    int* d_indexTrans,
//    vector3T* d_sorted_positions,
//    T* d_sorted_masses,
//    vector3T* d_sorted_velocities,
//    T** d_channels,
//    int** d_adjPage)
//{
//    __shared__ T buffer[7][8][8][8];
//    int cellid = (7 * 8 * 8 * 8 + blockDim.x - 1) / blockDim.x;
//    for (int i = 0; i < cellid; ++i)
//        if (blockDim.x * i + threadIdx.x < 7 * 8 * 8 * 8)
//            *((&buffer[0][0][0][0]) + blockDim.x * i + threadIdx.x) = (T)0;
//    __syncthreads();
//
//    int pageid = d_targetPages[blockIdx.x] - 1;
//    cellid = d_block_offsets[pageid];
//    int relParid = 512 * (blockIdx.x - d_virtualPageOffsets[pageid]) + threadIdx.x;
//    int parid = cellid + relParid;
//
//    int laneid = threadIdx.x & 0x1f;
//    bool bBoundary;
//    if (relParid < d_block_offsets[pageid + 1] - d_block_offsets[pageid])
//    {
//        cellid = d_cellids[parid] - 1;
//        bBoundary = laneid == 0 || cellid + 1 != d_cellids[parid - 1];
//    }
//    else
//        bBoundary = true;
//
//    unsigned int mark = __ballot_sync(0xffffffff, bBoundary); // a bit-mask 
//    mark = __brev(mark);
//    unsigned int interval = min(__clz(mark << (laneid + 1)), 31 - laneid);
//    mark = interval;
//    for (int iter = 1; iter & 0x1f; iter <<= 1) {
//        int tmp = __shfl_down_sync(__activemask(), mark, iter);
//        mark = tmp > mark ? tmp : mark; /*if (tmp > mark) mark = tmp;*/
//    }
//    mark = __shfl_sync(0xffffffff, mark, 0);
//    __syncthreads();
//
//    int smallest_node[3];
//    if (relParid < d_block_offsets[pageid + 1] - d_block_offsets[pageid]) {
//        T wOneD[3][3], wgOneD[3][3];
//
//        smallest_node[0] = smallest_nodes[cellid].x;
//        smallest_node[1] = smallest_nodes[cellid].y;
//        smallest_node[2] = smallest_nodes[cellid].z;
//
//
//        T dp0[3];
//        dp0[0] = (d_sorted_positions[parid].x - smallest_node[0] * dx) * one_over_dx;
//        dp0[1] = (d_sorted_positions[parid].y - smallest_node[1] * dx) * one_over_dx;
//        dp0[2] = (d_sorted_positions[parid].z - smallest_node[2] * dx) * one_over_dx;
//
//       
//        for (int v = 0; v < 3; ++v) {
//            T d0 = dp0[v];
//            T z = ((T)1.5 - d0);
//            wOneD[v][0] = (T)0.5 * z * z;
//            wgOneD[v][0] = -z;
//            d0 = d0 - 1;
//            wOneD[v][1] = (T)0.75 - d0 * d0;
//            wgOneD[v][1] = -d0 * 2;
//            z = (T)1.5 - (1 - d0);
//            wOneD[v][2] = (T)0.5 * z * z;
//            wgOneD[v][2] = z;
//        }
//
//        wgOneD[0][0] *= one_over_dx;
//        wgOneD[0][1] *= one_over_dx;
//        wgOneD[0][2] *= one_over_dx;
//        wgOneD[1][0] *= one_over_dx;
//        wgOneD[1][1] *= one_over_dx;
//        wgOneD[1][2] *= one_over_dx;
//        wgOneD[2][0] *= one_over_dx;
//        wgOneD[2][1] *= one_over_dx;
//        wgOneD[2][2] *= one_over_dx;
//
//        T vel[3];
//        int parid_mapped = d_indexTrans[parid];
//        vel[0] = d_sorted_velocities[parid_mapped].x;
//        vel[1] = d_sorted_velocities[parid_mapped].y;
//        vel[2] = d_sorted_velocities[parid_mapped].z;
//
//        smallest_node[0] = smallest_node[0] & 0x3;
//        smallest_node[1] = smallest_node[1] & 0x3;
//        smallest_node[2] = smallest_node[2] & 0x3;
//
//        T sig[9];
//        sig[0] = d_sigma[parid_mapped + (0) * numParticle]; sig[1] = d_sigma[parid_mapped + (1) * numParticle]; sig[2] = d_sigma[parid_mapped + (2) * numParticle];
//        sig[3] = d_sigma[parid_mapped + (3) * numParticle]; sig[4] = d_sigma[parid_mapped + (4) * numParticle]; sig[5] = d_sigma[parid_mapped + (5) * numParticle];
//        sig[6] = d_sigma[parid_mapped + (6) * numParticle]; sig[7] = d_sigma[parid_mapped + (7) * numParticle]; sig[8] = d_sigma[parid_mapped + (8) * numParticle];
//
//        T val[7];
//        T mass = d_sorted_masses[d_indices[parid]];
//
//        for (int i = 0; i < 3; ++i) {
//            for (int j = 0; j < 3; ++j) {
//                for (int k = 0; k < 3; ++k) {
//
//                    val[0] = mass * wOneD[0][i] * wOneD[1][j] * wOneD[2][k];
//                    val[1] = val[0] * vel[0];
//                    val[2] = val[0] * vel[1];
//                    val[3] = val[0] * vel[2];
//
//                    T wg[3];
//                    wg[0] = wgOneD[0][i] * wOneD[1][j] * wOneD[2][k];
//                    wg[1] = wOneD[0][i] * wgOneD[1][j] * wOneD[2][k];
//                    wg[2] = wOneD[0][i] * wOneD[1][j] * wgOneD[2][k];
//                    val[4] = -(sig[0] * wg[0] + sig[3] * wg[1] + sig[6] * wg[2]);
//                    val[5] = -(sig[1] * wg[0] + sig[4] * wg[1] + sig[7] * wg[2]);
//                    val[6] = -(sig[2] * wg[0] + sig[5] * wg[1] + sig[8] * wg[2]);
//
//                    for (int iter = 1; iter <= mark; iter <<= 1) {
//                        T tmp[7]; for (int i = 0; i < 7; ++i) tmp[i] = __shfl_down_sync(__activemask(), val[i], iter);
//                        if (interval >= iter) for (int i = 0; i < 7; ++i) val[i] += tmp[i];
//                    }
//
//                    if (bBoundary) for (int ii = 0; ii < 7; ++ii)
//                        atomicAdd(&(buffer[ii][smallest_node[0] + i][smallest_node[1] + j][smallest_node[2] + k]), val[ii]);
//
//                }
//            }
//        }
//    }
//    __syncthreads();
//
//    int block = threadIdx.x & 0x3f;
//    int ci = block >> 4;
//    int cj = (block & 0xc) >> 2;
//    int ck = block & 3;
//
//    block = threadIdx.x >> 6;
//    int bi = block >> 2;
//    int bj = (block & 2) >> 1;
//    int bk = block & 1;
//
//    int page_idx = block ? d_adjPage[block - 1][pageid] : pageid;
//
//    for (int ii = 0; ii < 7; ++ii)
//        if (buffer[ii][bi * 4 + ci][bj * 4 + cj][bk * 4 + ck] != 0)
//            atomicAdd((T*)((unsigned long long)d_channels[ii] + page_idx * 4096) + (ci * 16 + cj * 4 + ck), buffer[ii][bi * 4 + ci][bj * 4 + cj][bk * 4 + ck]);
//
//}


__global__ void P2G_APIC(
    const int numParticle,
    const int* d_targetPages,
    const int* d_virtualPageOffsets,
    const int3* smallest_nodes,
    const T* d_sigma,
    int* d_block_offsets,
    int* d_cellids,
    int* d_indices,
    int* d_indexTrans,
    vector3T* d_sorted_positions,
    T* d_sorted_C,
    T* d_sorted_masses,
    vector3T* d_sorted_velocities,
    T* d_B,
    T** d_channels,
    int** d_adjPage)
{
    __shared__ T buffer[10][8][8][8];
    int cellid = (10 * 8 * 8 * 8 + blockDim.x - 1) / blockDim.x;
    for (int i = 0; i < cellid; ++i)
        if (blockDim.x * i + threadIdx.x < 10 * 8 * 8 * 8)
            *((&buffer[0][0][0][0]) + blockDim.x * i + threadIdx.x) = (T)0;
    __syncthreads();

    int pageid = d_targetPages[blockIdx.x] - 1;
    cellid = d_block_offsets[pageid];
    int relParid = 512 * (blockIdx.x - d_virtualPageOffsets[pageid]) + threadIdx.x;
    int parid = cellid + relParid;

    int laneid = threadIdx.x & 0x1f;
    bool bBoundary;
    if (relParid < d_block_offsets[pageid + 1] - d_block_offsets[pageid])
    {
        cellid = d_cellids[parid] - 1;
        bBoundary = laneid == 0 || cellid + 1 != d_cellids[parid - 1];
    }
    else
        bBoundary = true;

    unsigned int mark = __ballot_sync(0xffffffff, bBoundary); // a bit-mask 
    mark = __brev(mark);
    unsigned int interval = min(__clz(mark << (laneid + 1)), 31 - laneid);
    mark = interval;
    for (int iter = 1; iter & 0x1f; iter <<= 1) {
        int tmp = __shfl_down_sync(__activemask(), mark, iter);
        mark = tmp > mark ? tmp : mark; /*if (tmp > mark) mark = tmp;*/
    }
    mark = __shfl_sync(0xffffffff, mark, 0);
    __syncthreads();

    int smallest_node[3];

    if (relParid < d_block_offsets[pageid + 1] - d_block_offsets[pageid]) {
        T wOneD[3][3], wgOneD[3][3];

        smallest_node[0] = smallest_nodes[cellid].x;
        smallest_node[1] = smallest_nodes[cellid].y;
        smallest_node[2] = smallest_nodes[cellid].z;

        float sig[9];
        int parid_trans = d_indexTrans[parid];
        sig[0] = d_sigma[parid_trans + (0) * numParticle]; sig[1] = d_sigma[parid_trans + (1) * numParticle]; sig[2] = d_sigma[parid_trans + (2) * numParticle];
        sig[3] = d_sigma[parid_trans + (3) * numParticle]; sig[4] = d_sigma[parid_trans + (4) * numParticle]; sig[5] = d_sigma[parid_trans + (5) * numParticle];
        sig[6] = d_sigma[parid_trans + (6) * numParticle]; sig[7] = d_sigma[parid_trans + (7) * numParticle]; sig[8] = d_sigma[parid_trans + (8) * numParticle];

        float B[9];
        B[0] = d_B[parid_trans + 0 * numParticle]; B[1] = d_B[parid_trans + 1 * numParticle]; B[2] = d_B[parid_trans + 2 * numParticle];
        B[3] = d_B[parid_trans + 3 * numParticle]; B[4] = d_B[parid_trans + 4 * numParticle]; B[5] = d_B[parid_trans + 5 * numParticle];
        B[6] = d_B[parid_trans + 6 * numParticle]; B[7] = d_B[parid_trans + 7 * numParticle]; B[8] = d_B[parid_trans + 8 * numParticle];
        for (int i = 0; i < 9; ++i)
            B[i] *= D_inverse;

        T xp[3];
        xp[0] = d_sorted_positions[parid].x - smallest_node[0] * dx;
        xp[1] = d_sorted_positions[parid].y - smallest_node[1] * dx;
        xp[2] = d_sorted_positions[parid].z - smallest_node[2] * dx;

        for (int v = 0; v < 3; ++v) {
            T d0 = xp[v] * one_over_dx;
            T z = ((T)1.5 - d0);
            wOneD[v][0] = (T)0.5 * z * z;
            wgOneD[v][0] = -z;
            d0 = d0 - 1;
            wOneD[v][1] = (T)0.75 - d0 * d0;
            wgOneD[v][1] = -d0 * 2;
            z = (T)1.5 - (1 - d0);
            wOneD[v][2] = (T)0.5 * z * z;
            wgOneD[v][2] = z;
        }

        wgOneD[0][0] *= one_over_dx;
        wgOneD[0][1] *= one_over_dx;
        wgOneD[0][2] *= one_over_dx;
        wgOneD[1][0] *= one_over_dx;
        wgOneD[1][1] *= one_over_dx;
        wgOneD[1][2] *= one_over_dx;
        wgOneD[2][0] *= one_over_dx;
        wgOneD[2][1] *= one_over_dx;
        wgOneD[2][2] *= one_over_dx;

        T vel[3];
        vel[0] = d_sorted_velocities[parid_trans].x;
        vel[1] = d_sorted_velocities[parid_trans].y;
        vel[2] = d_sorted_velocities[parid_trans].z;

        smallest_node[0] = smallest_node[0] & 0x3;
        smallest_node[1] = smallest_node[1] & 0x3;
        smallest_node[2] = smallest_node[2] & 0x3;

        T mass = d_sorted_masses[d_indices[parid]];
        T C = d_sorted_C[parid_trans];

        float val[10];// for (int i = 0; i < 10; i++) val[i] = 0.;
        T wg[3];
        T xi_minus_xp[3];
        T tmp[10];
        T weight;
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                for (int k = 0; k < 3; ++k) {

                    weight = wOneD[0][i] * wOneD[1][j] * wOneD[2][k];
                    wg[0] = wgOneD[0][i] * wOneD[1][j] * wOneD[2][k];
                    wg[1] = wOneD[0][i] * wgOneD[1][j] * wOneD[2][k];
                    wg[2] = wOneD[0][i] * wOneD[1][j] * wgOneD[2][k];

                    val[0] = mass * weight;
                    val[4] = -(sig[0] * wg[0] + sig[3] * wg[1] + sig[6] * wg[2]);
                    val[5] = -(sig[1] * wg[0] + sig[4] * wg[1] + sig[7] * wg[2]);
                    val[6] = -(sig[2] * wg[0] + sig[5] * wg[1] + sig[8] * wg[2]);
                    val[7] = weight * C;
                    val[8] = weight;
                    val[9] = 0.01f;// wOneD[0][i] * wOneD[1][j] * wOneD[2][k];


                    xi_minus_xp[0] = i * dx - xp[0];
                    xi_minus_xp[1] = j * dx - xp[1];
                    xi_minus_xp[2] = k * dx - xp[2];


                    val[1] = vel[0];
                    val[2] = vel[1];
                    val[3] = vel[2];
                    val[1] += (B[0] * xi_minus_xp[0] + B[3] * xi_minus_xp[1] + B[6] * xi_minus_xp[2]);
                    val[2] += (B[1] * xi_minus_xp[0] + B[4] * xi_minus_xp[1] + B[7] * xi_minus_xp[2]);
                    val[3] += (B[2] * xi_minus_xp[0] + B[5] * xi_minus_xp[1] + B[8] * xi_minus_xp[2]);
                    val[1] *= val[0];
                    val[2] *= val[0];
                    val[3] *= val[0];

                    //volatile T* vol_sh_max = sha_partialMax2

                    //for (int i = 0; i < 10; i++) {
                    //	for (int iter = 1; iter <= mark; iter <<= 1) {
                    //		tmp = __shfl_down_sync(__activemask(), val[i], iter);
                    //		if (interval >= iter) {
                    //			val[i] += tmp;
                    //		}
                    //	}
                    //}


                    for (int iter = 1; iter <= mark; iter <<= 1) {
                        for (int i = 0; i < 10; ++i) tmp[i] = __shfl_down_sync(__activemask(), val[i], iter);
                        if (interval >= iter) for (int i = 0; i < 10; ++i) val[i] += tmp[i];
                    }

                    if (bBoundary) for (int ii = 0; ii < 10; ++ii)
                        atomicAdd(&(buffer[ii][smallest_node[0] + i][smallest_node[1] + j][smallest_node[2] + k]), val[ii]);

                }
            }
        }
    }
    __syncthreads();

    int block = threadIdx.x & 0x3f;
    int ci = block >> 4;
    int cj = (block & 0xc) >> 2;
    int ck = block & 3;

    block = threadIdx.x >> 6;
    int bi = block >> 2;
    int bj = (block & 2) >> 1;
    int bk = block & 1;

    int page_idx = block ? d_adjPage[block - 1][pageid] : pageid;

    for (int ii = 0; ii < 10; ++ii)
        if (buffer[ii][bi * 4 + ci][bj * 4 + cj][bk * 4 + ck] != 0)
            atomicAdd((T*)((unsigned long long)d_channels[ii] + page_idx * 8192) + (ci * 16 + cj * 4 + ck), buffer[ii][bi * 4 + ci][bj * 4 + cj][bk * 4 + ck]);
}


__global__ void volP2G_APIC(
    const int numParticle,
    const int* d_targetPages,
    const int* d_virtualPageOffsets,
    const int3* smallest_nodes,

    int* d_block_offsets,
    int* d_cellids,
    int* d_indices,
    int* d_indexTrans,
    vector3T* d_sorted_positions,
    T* d_sorted_vol,
    T** d_channels,
    int** d_adjPage,
    T dt,
    T parabolic_M)
{
    __shared__ T buffer[2][8][8][8];
    

    int pageid = d_targetPages[blockIdx.x] - 1;

    int block = threadIdx.x & 0x3f;
    int ci = block >> 4;
    int cj = (block & 0xc) >> 2;
    int ck = block & 3;

    block = threadIdx.x >> 6;
    int bi = block >> 2;
    int bj = (block & 2) >> 1;
    int bk = block & 1;
    
    int page_idx = block ? d_adjPage[block - 1][pageid] : pageid;

    int cellid = (8 * 8 * 8 + blockDim.x - 1) / blockDim.x;
    for (int i = 0; i < cellid; ++i)
        if (blockDim.x * i + threadIdx.x < 8 * 8 * 8)
            *((&buffer[0][0][0][0]) + blockDim.x * i + threadIdx.x) = (T)0;
    
    buffer[1][bi * 4 + ci][bj * 4 + cj][bk * 4 + ck] =
        *((T*)((unsigned long long)d_channels[7] + (int)page_idx * 4096) + (ci * 16 + cj * 4 + ck));

    __syncthreads();

    
    cellid = d_block_offsets[pageid];
    int relParid = 512 * (blockIdx.x - d_virtualPageOffsets[pageid]) + threadIdx.x;
    int parid = cellid + relParid;

    int laneid = threadIdx.x & 0x1f;
    bool bBoundary;
    if (relParid < d_block_offsets[pageid + 1] - d_block_offsets[pageid])
    {
        cellid = d_cellids[parid] - 1;
        bBoundary = laneid == 0 || cellid + 1 != d_cellids[parid - 1];
    }
    else
        bBoundary = true;

    unsigned int mark = __ballot_sync(0xffffffff, bBoundary); // a bit-mask 
    mark = __brev(mark);
    unsigned int interval = min(__clz(mark << (laneid + 1)), 31 - laneid);
    mark = interval;
    for (int iter = 1; iter & 0x1f; iter <<= 1) {
        int tmp = __shfl_down_sync(__activemask(), mark, iter);
        mark = tmp > mark ? tmp : mark; /*if (tmp > mark) mark = tmp;*/
    }
    mark = __shfl_sync(0xffffffff, mark, 0);
    __syncthreads();

    int smallest_node[3];
    
    if (relParid < d_block_offsets[pageid + 1] - d_block_offsets[pageid]) {
        T wOneD[3][3], wgOneD[3][3];

        smallest_node[0] = smallest_nodes[cellid].x;
        smallest_node[1] = smallest_nodes[cellid].y;
        smallest_node[2] = smallest_nodes[cellid].z;

        int parid_trans = d_indexTrans[parid];

        T xp[3];
        xp[0] = d_sorted_positions[parid].x - smallest_node[0] * dx;
        xp[1] = d_sorted_positions[parid].y - smallest_node[1] * dx;
        xp[2] = d_sorted_positions[parid].z - smallest_node[2] * dx;

        for (int v = 0; v < 3; ++v) {
            T d0 = xp[v] * one_over_dx;
            T z = ((T)1.5 - d0);
            wOneD[v][0] = (T)0.5 * z * z;
            d0 = d0 - 1;
            wOneD[v][1] = (T)0.75 - d0 * d0;
            z = (T)1.5 - (1 - d0);
            wOneD[v][2] = (T)0.5 * z * z;
        }

        smallest_node[0] = smallest_node[0] & 0x3;
        smallest_node[1] = smallest_node[1] & 0x3;
        smallest_node[2] = smallest_node[2] & 0x3;

        T p_vol = d_sorted_vol[parid_trans];
        //T val;
		T val;
		T g_c;
		T tmp;
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                for (int k = 0; k < 3; ++k) {

                    g_c = buffer[1][smallest_node[0] + i][smallest_node[1] + j][smallest_node[2] + k];

                    val = p_vol * wOneD[0][i] * wOneD[1][j] * wOneD[2][k] * (g_c / dt + parabolic_M);
                    //volatile T* vol_sh_max = sha_partialMax2
                    for (int iter = 1; iter <= mark; iter <<= 1) {
                        tmp = __shfl_down_sync(__activemask(), val, iter);
                        if (interval >= iter) val += tmp;
                    }
                    if (bBoundary)
                        atomicAdd(&(buffer[0][smallest_node[0] + i][smallest_node[1] + j][smallest_node[2] + k]), val);
                }
            }
        }
    }
    __syncthreads();

    
    if (buffer[0][bi * 4 + ci][bj * 4 + cj][bk * 4 + ck] != 0)
        atomicAdd((T*)((unsigned long long)d_channels[10] + page_idx * 4096) + (ci * 16 + cj * 4 + ck), buffer[0][bi * 4 + ci][bj * 4 + cj][bk * 4 + ck]);
}

__global__ void preConditionP2G_APIC(
    const int numParticle,
    const int* d_targetPages,
    const int* d_virtualPageOffsets,
    const int3* smallest_nodes,

    int* d_block_offsets,
    int* d_cellids,
    int* d_indices,
    int* d_indexTrans,
    vector3T* d_sorted_positions,
    T* d_sorted_vol,
    T* d_sorted_FP,
    T** d_channels,
    int** d_adjPage,
    T dt,
    T parabolic_M)
{
    __shared__ T buffer[8][8][8];

    int cellid = (8 * 8 * 8 + blockDim.x - 1) / blockDim.x;
    for (int i = 0; i < cellid; ++i)
        if (blockDim.x * i + threadIdx.x < 8 * 8 * 8)
            *((&buffer[0][0][0]) + blockDim.x * i + threadIdx.x) = (T)0;

    __syncthreads();

    int pageid = d_targetPages[blockIdx.x] - 1;

    cellid = d_block_offsets[pageid];
    int relParid = 512 * (blockIdx.x - d_virtualPageOffsets[pageid]) + threadIdx.x;
    int parid = cellid + relParid;

    int laneid = threadIdx.x & 0x1f;
    bool bBoundary;
    if (relParid < d_block_offsets[pageid + 1] - d_block_offsets[pageid])
    {
        cellid = d_cellids[parid] - 1;
        bBoundary = laneid == 0 || cellid + 1 != d_cellids[parid - 1];
    }
    else
        bBoundary = true;

    unsigned int mark = __ballot_sync(0xffffffff, bBoundary); // a bit-mask 
    mark = __brev(mark);
    unsigned int interval = min(__clz(mark << (laneid + 1)), 31 - laneid);
    mark = interval;
    for (int iter = 1; iter & 0x1f; iter <<= 1) {
        int tmp = __shfl_down_sync(__activemask(), mark, iter);
        mark = tmp > mark ? tmp : mark; /*if (tmp > mark) mark = tmp;*/
    }
    mark = __shfl_sync(0xffffffff, mark, 0);
    __syncthreads();

    int smallest_node[3];
    
    if (relParid < d_block_offsets[pageid + 1] - d_block_offsets[pageid]) {
        T wOneD[3][3], wgOneD[3][3];

        smallest_node[0] = smallest_nodes[cellid].x;
        smallest_node[1] = smallest_nodes[cellid].y;
        smallest_node[2] = smallest_nodes[cellid].z;

        int parid_trans = d_indexTrans[parid];

        T xp[3];
        xp[0] = d_sorted_positions[parid].x - smallest_node[0] * dx;
        xp[1] = d_sorted_positions[parid].y - smallest_node[1] * dx;
        xp[2] = d_sorted_positions[parid].z - smallest_node[2] * dx;

        for (int v = 0; v < 3; ++v) {
            T d0 = xp[v] * one_over_dx;
            T z = ((T)1.5 - d0);
            wOneD[v][0] = (T)0.5 * z * z;
            d0 = d0 - 1;
            wOneD[v][1] = (T)0.75 - d0 * d0;
            z = (T)1.5 - (1 - d0);
            wOneD[v][2] = (T)0.5 * z * z;
        }

        smallest_node[0] = smallest_node[0] & 0x3;
        smallest_node[1] = smallest_node[1] & 0x3;
        smallest_node[2] = smallest_node[2] & 0x3;

        T vol = d_sorted_vol[parid_trans];
        T FP = d_sorted_FP[parid_trans];
		T val;// = 0.f;
		T tmp;
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                for (int k = 0; k < 3; ++k) {
                    val = vol * wOneD[0][i] * wOneD[1][j] * wOneD[2][k] * (FP * parabolic_M + (1.f / dt));
                    //volatile T* vol_sh_max = sha_partialMax2
                    for (int iter = 1; iter <= mark; iter <<= 1) {
                        tmp = __shfl_down_sync(__activemask(), val, iter);
                        if (interval >= iter) val += tmp;
                    }
                    if (bBoundary)
                        atomicAdd(&(buffer[smallest_node[0] + i][smallest_node[1] + j][smallest_node[2] + k]), val);
                }
            }
        }
    }
    __syncthreads();


    int block = threadIdx.x & 0x3f;
    int ci = block >> 4;
    int cj = (block & 0xc) >> 2;
    int ck = block & 3;

    block = threadIdx.x >> 6;
    int bi = block >> 2;
    int bj = (block & 2) >> 1;
    int bk = block & 1;

    int page_idx = block ? d_adjPage[block - 1][pageid] : pageid;

    if (buffer[bi * 4 + ci][bj * 4 + cj][bk * 4 + ck] != 0)
        atomicAdd((T*)((unsigned long long)d_channels[11] + page_idx * 4096) + (ci * 16 + cj * 4 + ck), buffer[bi * 4 + ci][bj * 4 + cj][bk * 4 + ck]);
}




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
    T parabolic_M)
{
    __shared__ T buffer[2][8][8][8];
    int cellid = (8 * 8 * 8 + blockDim.x - 1) / blockDim.x;
    for (int i = 0; i < cellid; ++i)
        if (blockDim.x * i + threadIdx.x < 8 * 8 * 8)
            *((&buffer[0][0][0][0]) + blockDim.x * i + threadIdx.x) = (T)0;



    int pageid = d_targetPages[blockIdx.x] - 1;

    int block = threadIdx.x & 0x3f;
    int ci = block >> 4;
    int cj = (block & 0xc) >> 2;
    int ck = block & 3;

    block = threadIdx.x >> 6;
    int bi = block >> 2;
    int bj = (block & 2) >> 1;
    int bk = block & 1;

    int page_idx = block ? d_adjPage[block - 1][pageid] : pageid;

    buffer[1][bi * 4 + ci][bj * 4 + cj][bk * 4 + ck] = d_channels[page_idx * 64 + (ci * 16 + cj * 4 + ck)];

    __syncthreads();


    cellid = d_block_offsets[pageid];
    int relParid = 512 * (blockIdx.x - d_virtualPageOffsets[pageid]) + threadIdx.x;
    int parid = cellid + relParid;

    int laneid = threadIdx.x & 0x1f;
    bool bBoundary;
    if (relParid < d_block_offsets[pageid + 1] - d_block_offsets[pageid])
    {
        cellid = d_cellids[parid] - 1;
        bBoundary = laneid == 0 || cellid + 1 != d_cellids[parid - 1];
    }
    else
        bBoundary = true;

    unsigned int mark = __ballot_sync(0xffffffff, bBoundary); // a bit-mask 
    mark = __brev(mark);
    unsigned int interval = min(__clz(mark << (laneid + 1)), 31 - laneid);
    mark = interval;
    for (int iter = 1; iter & 0x1f; iter <<= 1) {
        int tmp = __shfl_down_sync(__activemask(), mark, iter);
        mark = tmp > mark ? tmp : mark; /*if (tmp > mark) mark = tmp;*/
    }
    mark = __shfl_sync(0xffffffff, mark, 0);
    __syncthreads();

    int smallest_node[3];
    
    if (relParid < d_block_offsets[pageid + 1] - d_block_offsets[pageid]) {
        T wOneD[3][3], wgOneD[3][3];

        smallest_node[0] = smallest_nodes[cellid].x;
        smallest_node[1] = smallest_nodes[cellid].y;
        smallest_node[2] = smallest_nodes[cellid].z;

        int parid_trans = d_indexTrans[parid];


        T xp[3];
        xp[0] = d_sorted_positions[parid].x - smallest_node[0] * dx;
        xp[1] = d_sorted_positions[parid].y - smallest_node[1] * dx;
        xp[2] = d_sorted_positions[parid].z - smallest_node[2] * dx;

        for (int v = 0; v < 3; ++v) {
            T d0 = xp[v] * one_over_dx;
            T z = ((T)1.5 - d0);
            wOneD[v][0] = (T)0.5 * z * z;
            wgOneD[v][0] = -z;
            d0 = d0 - 1;
            wOneD[v][1] = (T)0.75 - d0 * d0;
            wgOneD[v][1] = -d0 * 2;
            z = (T)1.5 - (1 - d0);
            wOneD[v][2] = (T)0.5 * z * z;
            wgOneD[v][2] = z;
        }

        wgOneD[0][0] *= one_over_dx;
        wgOneD[0][1] *= one_over_dx;
        wgOneD[0][2] *= one_over_dx;
        wgOneD[1][0] *= one_over_dx;
        wgOneD[1][1] *= one_over_dx;
        wgOneD[1][2] *= one_over_dx;
        wgOneD[2][0] *= one_over_dx;
        wgOneD[2][1] *= one_over_dx;
        wgOneD[2][2] *= one_over_dx;



        smallest_node[0] = smallest_node[0] & 0x3;
        smallest_node[1] = smallest_node[1] & 0x3;
        smallest_node[2] = smallest_node[2] & 0x3;

        T p_vol = d_sorted_vol[parid_trans];
        T p_fp = d_FP[parid_trans];
        vector3T col = d_sorted_col[parid_trans];

		T val;// = 0.f;
		T wg[3];
		T HW;
		T g_x;
		T tmp;
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                for (int k = 0; k < 3; ++k) {

                    
                    wg[0] = wgOneD[0][i] * wOneD[1][j] * wOneD[2][k];
                    wg[1] = wOneD[0][i] * wgOneD[1][j] * wOneD[2][k];
                    wg[2] = wOneD[0][i] * wOneD[1][j] * wgOneD[2][k];

                    HW = wg[0] * col.x + wg[1] * col.y + wg[2] * col.z;

                    g_x = buffer[1][smallest_node[0] + i][smallest_node[1] + j][smallest_node[2] + k];

                    val = g_x * (p_vol * wOneD[0][i] * wOneD[1][j] * wOneD[2][k] * (p_fp * parabolic_M + 1.f / dt) + HW * parabolic_M);
                    //volatile T* vol_sh_max = sha_partialMax2
                    for (int iter = 1; iter <= mark; iter <<= 1) {
                        tmp = __shfl_down_sync(__activemask(), val, iter);
                        if (interval >= iter)  val += tmp;
                    }

                    if (bBoundary)
                        atomicAdd(&(buffer[0][smallest_node[0] + i][smallest_node[1] + j][smallest_node[2] + k]), val);

                }
            }
        }
    }
    __syncthreads();


    if (buffer[0][bi * 4 + ci][bj * 4 + cj][bk * 4 + ck] != 0)
        atomicAdd(&d_ax[page_idx * 64 + (ci * 16 + cj * 4 + ck)], buffer[0][bi * 4 + ci][bj * 4 + cj][bk * 4 + ck]);
}

