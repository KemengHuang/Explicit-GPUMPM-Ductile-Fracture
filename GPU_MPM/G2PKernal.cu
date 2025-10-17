#include "G2PKernal.cuh"

#include <cstdio>
#define LOG_NUM_BANKS	 4
#define CONFLICT_FREE_OFFSET(index) ((index) >> LOG_NUM_BANKS)

__device__ void matrixMatrixMultiplication(const float* a, const float* b, float* c)
{
    c[0] = a[0] * b[0] + a[3] * b[1] + a[6] * b[2];
    c[1] = a[1] * b[0] + a[4] * b[1] + a[7] * b[2];
    c[2] = a[2] * b[0] + a[5] * b[1] + a[8] * b[2];
    c[3] = a[0] * b[3] + a[3] * b[4] + a[6] * b[5];
    c[4] = a[1] * b[3] + a[4] * b[4] + a[7] * b[5];
    c[5] = a[2] * b[3] + a[5] * b[4] + a[8] * b[5];
    c[6] = a[0] * b[6] + a[3] * b[7] + a[6] * b[8];
    c[7] = a[1] * b[6] + a[4] * b[7] + a[7] * b[8];
    c[8] = a[2] * b[6] + a[5] * b[7] + a[8] * b[8];
}

__device__ void matrixMatrixMultiplication(const double* a, const double* b, double* c)
{
    c[0] = a[0] * b[0] + a[3] * b[1] + a[6] * b[2];
    c[1] = a[1] * b[0] + a[4] * b[1] + a[7] * b[2];
    c[2] = a[2] * b[0] + a[5] * b[1] + a[8] * b[2];
    c[3] = a[0] * b[3] + a[3] * b[4] + a[6] * b[5];
    c[4] = a[1] * b[3] + a[4] * b[4] + a[7] * b[5];
    c[5] = a[2] * b[3] + a[5] * b[4] + a[8] * b[5];
    c[6] = a[0] * b[6] + a[3] * b[7] + a[6] * b[8];
    c[7] = a[1] * b[6] + a[4] * b[7] + a[7] * b[8];
    c[8] = a[2] * b[6] + a[5] * b[7] + a[8] * b[8];
}
    __global__ void G2P_MLS(
        const int numParticle,
        const int* d_targetPages,
        const int* d_virtualPageOffsets,
        const int3* smallest_nodes,
        int* d_block_offsets,
        int* d_cellids,
        int* d_indices,
        int* d_indexTrans,
        vector3T* d_sorted_positions,
		T* d_sorted_C,
		T* d_sorted_C_sort,
        vector3T* d_sorted_velocities,
        T** d_channels,
        T* d_sorted_F,
        T* d_B,
        T* d_tmp,
        T dt,
        int** d_adjPage
    ) {
        __shared__ T buffer[4][8][8][8];

        int pageid = d_targetPages[blockIdx.x] - 1; // from virtual to physical page
        int cellid = d_block_offsets[pageid]; // 
        int relParid = 512 * (blockIdx.x - d_virtualPageOffsets[pageid]) + threadIdx.x;
        int parid = cellid + relParid;

        int block = threadIdx.x & 0x3f;
        int ci = block >> 4;
        int cj = (block & 0xc) >> 2;
        int ck = block & 3;

        block = threadIdx.x >> 6;
        int bi = block >> 2;
        int bj = (block & 2) >> 1;
        int bk = block & 1;

        int page_idx = block ? d_adjPage[block - 1][pageid] : pageid;

        // vel
        for (int v = 0; v < 3; ++v)
            buffer[v][bi * 4 + ci][bj * 4 + cj][bk * 4 + ck] =
            *((T*)((unsigned long long)d_channels[1 + v] + (int)page_idx * 8192) + (ci * 16 + cj * 4 + ck));
		buffer[3][bi * 4 + ci][bj * 4 + cj][bk * 4 + ck] =
			*((T*)((unsigned long long)d_channels[7] + (int)page_idx * 8192) + (ci * 16 + cj * 4 + ck));
        __syncthreads();

        int smallest_node[3];
        if (relParid < d_block_offsets[pageid + 1] - d_block_offsets[pageid]) {
            cellid = d_cellids[parid] - 1;
            T wOneD[3][3];

            smallest_node[0] = smallest_nodes[cellid].x;
            smallest_node[1] = smallest_nodes[cellid].y;
            smallest_node[2] = smallest_nodes[cellid].z;

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

            int c = 0;
            T tmp[27];
            for (int i = 0; i < 3; ++i) {
                for (int j = 0; j < 3; ++j) {
                    for (int k = 0; k < 3; ++k) {
                        tmp[c++] = wOneD[0][i] * wOneD[1][j] * wOneD[2][k];
                    }
                }
            }

            for (int v = 0; v < 3; ++v)
                smallest_node[v] = smallest_node[v] & 0x3;

            T val[9]; for (int i = 0; i < 4; ++i) val[i] = 0.f;

            c = 0;
            for (int i = 0; i < 3; ++i) {
                for (int j = 0; j < 3; ++j) {
                    for (int k = 0; k < 3; ++k) {
                        // v_pic

                        val[0] += tmp[c] * buffer[0][smallest_node[0] + i][smallest_node[1] + j][smallest_node[2] + k];
                        val[1] += tmp[c] * buffer[1][smallest_node[0] + i][smallest_node[1] + j][smallest_node[2] + k];
                        val[2] += tmp[c] * buffer[2][smallest_node[0] + i][smallest_node[1] + j][smallest_node[2] + k];
						val[3] += tmp[c++] * buffer[3][smallest_node[0] + i][smallest_node[1] + j][smallest_node[2] + k];
                    }
                }
            }


			T norms = sqrt(val[0] * val[0] + val[1] * val[1] + val[2] * val[2]);
			if (norms > 11.137f) {
				for (int i = 0; i < 3; i++) {
					T vi = val[i];

					val[i] = vi / norms * 11.137f;

				}
			}



            d_sorted_velocities[parid].x = val[0];
            d_sorted_velocities[parid].y = val[1];
            d_sorted_velocities[parid].z = val[2];

			int parid_trans = d_indexTrans[parid];

			T p_c = d_sorted_C[parid_trans];// = __max(val[3], 0);
			T new_c = p_c + val[3];

			d_sorted_C_sort[parid] = __max(__min(p_c, new_c), 0);

            /*d_tmp[parid] = val[0];
            d_tmp[parid + numParticle] = val[1];
            d_tmp[parid + numParticle * 2] = val[2];*/

            d_sorted_positions[parid].x += val[0] * dt;
            d_sorted_positions[parid].y += val[1] * dt;
            d_sorted_positions[parid].z += val[2] * dt;

            for (int i = 0; i < 9; ++i) val[i] = 0.f;

            c = 0;
            for (int i = 0; i < 3; ++i) {
                for (int j = 0; j < 3; ++j) {
                    for (int k = 0; k < 3; ++k) {
                        // B
                        val[0] += tmp[c] * buffer[0][smallest_node[0] + i][smallest_node[1] + j][smallest_node[2] + k] * (i * dx - xp[0]);
                        val[1] += tmp[c] * buffer[1][smallest_node[0] + i][smallest_node[1] + j][smallest_node[2] + k] * (i * dx - xp[0]);
                        val[2] += tmp[c] * buffer[2][smallest_node[0] + i][smallest_node[1] + j][smallest_node[2] + k] * (i * dx - xp[0]);
                        val[3] += tmp[c] * buffer[0][smallest_node[0] + i][smallest_node[1] + j][smallest_node[2] + k] * (j * dx - xp[1]);
                        val[4] += tmp[c] * buffer[1][smallest_node[0] + i][smallest_node[1] + j][smallest_node[2] + k] * (j * dx - xp[1]);
                        val[5] += tmp[c] * buffer[2][smallest_node[0] + i][smallest_node[1] + j][smallest_node[2] + k] * (j * dx - xp[1]);
                        val[6] += tmp[c] * buffer[0][smallest_node[0] + i][smallest_node[1] + j][smallest_node[2] + k] * (k * dx - xp[2]);
                        val[7] += tmp[c] * buffer[1][smallest_node[0] + i][smallest_node[1] + j][smallest_node[2] + k] * (k * dx - xp[2]);
                        val[8] += tmp[c++] * buffer[2][smallest_node[0] + i][smallest_node[1] + j][smallest_node[2] + k] * (k * dx - xp[2]);
                    }
                }
            }

            for (int i = 0; i < 9; ++i) d_tmp[parid + (i) * numParticle] = val[i];

            for (int i = 0; i < 9; ++i) val[i] = val[i] * dt * D_inverse;
            val[0] += 1.f; val[4] += 1.f; val[8] += 1.f;

            T F[9];
            
            for (int i = 0; i < 9; ++i) F[i] = d_sorted_F[parid_trans + i * numParticle];

            T result[9];
            matrixMatrixMultiplication(&(val[0]), F, result);

            for (int i = 0; i < 9; ++i) d_tmp[parid + (i + 9) * numParticle] = result[i];
        }

    }

__global__ void G2P_APIC(
    const int numParticle, 
    const int* d_targetPages,
    const int* d_virtualPageOffsets, 
    const int3* smallest_nodes,
    int* d_block_offsets, 
    int* d_cellids, 
    int* d_indices, 
    int* d_indexTrans, 
    vector3T* d_sorted_positions, 
    T* d_sorted_C, 
    T* d_sorted_C_sort, 
    vector3T* d_sorted_velocities,
    T** d_channels, 
    T* d_sorted_F, 
    T* d_sorted_B, 
    T* d_tmp, 
    T dt, int** d_adjPage) {

    __shared__ T buffer[4][8][8][8];

    int pageid = d_targetPages[blockIdx.x] - 1; // from virtual to physical page
    int cellid = d_block_offsets[pageid]; // 
    int relParid = 512 * (blockIdx.x - d_virtualPageOffsets[pageid]) + threadIdx.x;
    int parid = cellid + relParid;

    int block = threadIdx.x & 0x3f;
    int ci = block >> 4;
    int cj = (block & 0xc) >> 2;
    int ck = block & 3;

    block = threadIdx.x >> 6;
    int bi = block >> 2;
    int bj = (block & 2) >> 1;
    int bk = block & 1;

    int page_idx = block ? d_adjPage[block - 1][pageid] : pageid;

    // vel
    for (int v = 0; v < 3; ++v)
        buffer[v][bi * 4 + ci][bj * 4 + cj][bk * 4 + ck] =
        *((T*)((unsigned long long)d_channels[1 + v] + (int)page_idx * 8192) + (ci * 16 + cj * 4 + ck));
    buffer[3][bi * 4 + ci][bj * 4 + cj][bk * 4 + ck] =
        *((T*)((unsigned long long)d_channels[7] + (int)page_idx * 8192) + (ci * 16 + cj * 4 + ck));

    __syncthreads();

    int smallest_node[3];
    if (relParid < d_block_offsets[pageid + 1] - d_block_offsets[pageid]) {
        cellid = d_cellids[parid] - 1;
        // quadratic B spline weights

        T wOneD[3][3], wgOneD[3][3];

        smallest_node[0] = smallest_nodes[cellid].x;
        smallest_node[1] = smallest_nodes[cellid].y;
        smallest_node[2] = smallest_nodes[cellid].z;

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

        for (int v = 0; v < 3; ++v) smallest_node[v] = smallest_node[v] & 0x3;

        T val[9]; for (int i = 0; i < 4; ++i) val[i] = 0.f;



		int c = 0;
        float tmp[27];
		for (int i = 0; i < 3; ++i) {
			for (int j = 0; j < 3; ++j) {
				for (int k = 0; k < 3; ++k) {
					tmp[c++] = wOneD[0][i] * wOneD[1][j] * wOneD[2][k];
				}
			}
		}


		c = 0;
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                for (int k = 0; k < 3; ++k) {
                    // v_pic


                    //if (!isnan(wOneD[0][i] * wOneD[1][j] * wOneD[2][k] * buffer[0][smallest_node[0] + i][smallest_node[1] + j][smallest_node[2] + k]) &&
                    //    !isnan(wOneD[0][i] * wOneD[1][j] * wOneD[2][k] * buffer[1][smallest_node[0] + i][smallest_node[1] + j][smallest_node[2] + k]) &&
                    //    !isnan(wOneD[0][i] * wOneD[1][j] * wOneD[2][k] * buffer[2][smallest_node[0] + i][smallest_node[1] + j][smallest_node[2] + k]) &&
                    //    !isnan(wOneD[0][i] * wOneD[1][j] * wOneD[2][k] * buffer[3][smallest_node[0] + i][smallest_node[1] + j][smallest_node[2] + k]) &&

                    //    !isinf(wOneD[0][i] * wOneD[1][j] * wOneD[2][k] * buffer[0][smallest_node[0] + i][smallest_node[1] + j][smallest_node[2] + k]) &&
                    //    !isinf(wOneD[0][i] * wOneD[1][j] * wOneD[2][k] * buffer[1][smallest_node[0] + i][smallest_node[1] + j][smallest_node[2] + k]) &&
                    //    !isinf(wOneD[0][i] * wOneD[1][j] * wOneD[2][k] * buffer[2][smallest_node[0] + i][smallest_node[1] + j][smallest_node[2] + k]))
                    //{
                        val[0] += tmp[c] * buffer[0][smallest_node[0] + i][smallest_node[1] + j][smallest_node[2] + k];
                        val[1] += tmp[c] * buffer[1][smallest_node[0] + i][smallest_node[1] + j][smallest_node[2] + k];
                        val[2] += tmp[c] * buffer[2][smallest_node[0] + i][smallest_node[1] + j][smallest_node[2] + k];
                        val[3] += tmp[c++] * buffer[3][smallest_node[0] + i][smallest_node[1] + j][smallest_node[2] + k];
                    //}
                }
            }
        }

        __syncthreads();
        //d_tmp[parid] = val[0];
        //d_tmp[parid + numParticle] = val[1];
        //d_tmp[parid + numParticle * 2] = val[2];

        int parid_trans = d_indexTrans[parid];

        //d_sorted_positions[parid].x += d_sorted_velocities[parid_trans].x * dt;
        //d_sorted_positions[parid].y += d_sorted_velocities[parid_trans].y * dt;
        //d_sorted_positions[parid].z += d_sorted_velocities[parid_trans].z * dt;


  //      T norms = sqrt(val[0] * val[0] + val[1] * val[1] + val[2] * val[2]);
		//if (norms > 11.137f) {
		//	for (int i = 0; i < 3; i++) {
  //              T vi = val[i];
		//		
		//		val[i] = vi / norms * 11.137f;

		//	}
		//}

        d_sorted_velocities[parid].x = val[0];
        d_sorted_velocities[parid].y = val[1];
        d_sorted_velocities[parid].z = val[2];
        
        T p_c = d_sorted_C[parid_trans];// = __max(val[3], 0);
        T new_c = p_c + val[3];

        d_sorted_C_sort[parid] = __max(__min(p_c, new_c), 0);

        d_sorted_positions[parid].x += val[0] * dt;
        d_sorted_positions[parid].y += val[1] * dt;
        d_sorted_positions[parid].z += val[2] * dt;

        for (int i = 0; i < 9; ++i) val[i] = 0.f;
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                for (int k = 0; k < 3; ++k) {
                    // F
                    val[0] += buffer[0][smallest_node[0] + i][smallest_node[1] + j][smallest_node[2] + k] * wgOneD[0][i] * wOneD[1][j] * wOneD[2][k];
                    val[1] += buffer[1][smallest_node[0] + i][smallest_node[1] + j][smallest_node[2] + k] * wgOneD[0][i] * wOneD[1][j] * wOneD[2][k];
                    val[2] += buffer[2][smallest_node[0] + i][smallest_node[1] + j][smallest_node[2] + k] * wgOneD[0][i] * wOneD[1][j] * wOneD[2][k];
                    val[3] += buffer[0][smallest_node[0] + i][smallest_node[1] + j][smallest_node[2] + k] * wOneD[0][i] * wgOneD[1][j] * wOneD[2][k];
                    val[4] += buffer[1][smallest_node[0] + i][smallest_node[1] + j][smallest_node[2] + k] * wOneD[0][i] * wgOneD[1][j] * wOneD[2][k];
                    val[5] += buffer[2][smallest_node[0] + i][smallest_node[1] + j][smallest_node[2] + k] * wOneD[0][i] * wgOneD[1][j] * wOneD[2][k];
                    val[6] += buffer[0][smallest_node[0] + i][smallest_node[1] + j][smallest_node[2] + k] * wOneD[0][i] * wOneD[1][j] * wgOneD[2][k];
                    val[7] += buffer[1][smallest_node[0] + i][smallest_node[1] + j][smallest_node[2] + k] * wOneD[0][i] * wOneD[1][j] * wgOneD[2][k];
                    val[8] += buffer[2][smallest_node[0] + i][smallest_node[1] + j][smallest_node[2] + k] * wOneD[0][i] * wOneD[1][j] * wgOneD[2][k];
                }
            }
        }

		__syncthreads();

        for (int i = 0; i < 9; ++i) val[i] = val[i] * dt;
        val[0] += 1.f;
        val[4] += 1.f;
        val[8] += 1.f;


        T F[9];
        
        
        for (int i = 0; i < 9; ++i) F[i] = d_sorted_F[parid_trans + i * numParticle];

        T result[9];
        matrixMatrixMultiplication(&(val[0]), F, result);


		//if (result[0] == 0.f) {
		//	result[0] = 1.f;
		//}
		//if (result[4] == 0.f) {
		//	result[4] = 1.f;
		//}
		//if (result[8] == 0.f) {
		//	result[8] = 1.f;
		//}

        for (int i = 0; i < 9; ++i) d_tmp[parid + (i) * numParticle] = result[i];
		c = 0;
        for (int i = 0; i < 9; ++i) val[i] = 0.f;
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                for (int k = 0; k < 3; ++k) {
                    // B
                    val[0] += tmp[c] * buffer[0][smallest_node[0] + i][smallest_node[1] + j][smallest_node[2] + k] * (i * dx - xp[0]);
                    val[1] += tmp[c] * buffer[1][smallest_node[0] + i][smallest_node[1] + j][smallest_node[2] + k] * (i * dx - xp[0]);
                    val[2] += tmp[c] * buffer[2][smallest_node[0] + i][smallest_node[1] + j][smallest_node[2] + k] * (i * dx - xp[0]);
                    val[3] += tmp[c] * buffer[0][smallest_node[0] + i][smallest_node[1] + j][smallest_node[2] + k] * (j * dx - xp[1]);
                    val[4] += tmp[c] * buffer[1][smallest_node[0] + i][smallest_node[1] + j][smallest_node[2] + k] * (j * dx - xp[1]);
                    val[5] += tmp[c] * buffer[2][smallest_node[0] + i][smallest_node[1] + j][smallest_node[2] + k] * (j * dx - xp[1]);
                    val[6] += tmp[c] * buffer[0][smallest_node[0] + i][smallest_node[1] + j][smallest_node[2] + k] * (k * dx - xp[2]);
                    val[7] += tmp[c] * buffer[1][smallest_node[0] + i][smallest_node[1] + j][smallest_node[2] + k] * (k * dx - xp[2]);
                    val[8] += tmp[c++] * buffer[2][smallest_node[0] + i][smallest_node[1] + j][smallest_node[2] + k] * (k * dx - xp[2]);

                }
            }
        }

        for (int i = 0; i < 9; ++i) d_tmp[parid + (i + 9) * numParticle] = val[i];
    }
}
    

//__global__ void G2P_FLIP(
//    const int numParticle,
//    const int* d_targetPages,
//    const int* d_virtualPageOffsets,
//    const int3* smallest_nodes,
//    int* d_block_offsets,
//    int* d_cellids,
//    int* d_indices,
//    int* d_indexTrans,
//    vector3T* d_sorted_positions,
//    vector3T* d_sorted_velocities,
//    T** d_channels,
//    T* d_sorted_F,
//    T* d_tmp,
//    T dt,
//    int** d_adjPage
//) {
//    const static T flip = 0.95f;
//
//    __shared__ T buffer[6][8][8][8];
//
//    int pageid = d_targetPages[blockIdx.x] - 1; // from virtual to physical page
//    int cellid = d_block_offsets[pageid]; // 
//    int relParid = 512 * (blockIdx.x - d_virtualPageOffsets[pageid]) + threadIdx.x;
//    int parid = cellid + relParid;
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
//    // vel0
//    for (int v = 0; v < 3; ++v)
//        buffer[v][bi * 4 + ci][bj * 4 + cj][bk * 4 + ck] =
//        *((T*)((unsigned long long)d_channels[7 + v] + (int)page_idx * 8192) + (ci * 16 + cj * 4 + ck));
//    // vel
//    for (int v = 0; v < 3; ++v)
//        buffer[v + 3][bi * 4 + ci][bj * 4 + cj][bk * 4 + ck] =
//        *((T*)((unsigned long long)d_channels[1 + v] + (int)page_idx * 8192) + (ci * 16 + cj * 4 + ck));
//
//    __syncthreads();
//
//    int smallest_node[3];
//    if (relParid < d_block_offsets[pageid + 1] - d_block_offsets[pageid]) {
//        cellid = d_cellids[parid] - 1;
//        // quadratic B spline weights
//
//        T wOneD[3][3], wgOneD[3][3];
//        smallest_node[0] = smallest_nodes[cellid].x;
//        smallest_node[1] = smallest_nodes[cellid].y;
//        smallest_node[2] = smallest_nodes[cellid].z;
//
//
//        T dp0[3];
//        dp0[0] = (d_sorted_positions[parid].x - (T)smallest_node[0] * dx) * one_over_dx;
//        dp0[1] = (d_sorted_positions[parid].y - (T)smallest_node[1] * dx) * one_over_dx;
//        dp0[2] = (d_sorted_positions[parid].z - (T)smallest_node[2] * dx) * one_over_dx;
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
//        for (int v = 0; v < 3; ++v) smallest_node[v] = smallest_node[v] & 0x3;
//
//        T val[9];
//        val[0] = 0.0f; val[1] = 0.0f; val[2] = 0.0f;
//        val[3] = 0.0f; val[4] = 0.0f; val[5] = 0.0f;
//
//        for (int i = 0; i < 3; ++i) {
//            for (int j = 0; j < 3; ++j) {
//                for (int k = 0; k < 3; ++k) {
//                    // v_diff
//                    val[0] += wOneD[0][i] * wOneD[1][j] * wOneD[2][k] * buffer[0][smallest_node[0] + i][smallest_node[1] + j][smallest_node[2] + k];
//                    val[1] += wOneD[0][i] * wOneD[1][j] * wOneD[2][k] * buffer[1][smallest_node[0] + i][smallest_node[1] + j][smallest_node[2] + k];
//                    val[2] += wOneD[0][i] * wOneD[1][j] * wOneD[2][k] * buffer[2][smallest_node[0] + i][smallest_node[1] + j][smallest_node[2] + k];
//                    // v_pic
//                    val[3] += wOneD[0][i] * wOneD[1][j] * wOneD[2][k] * buffer[3][smallest_node[0] + i][smallest_node[1] + j][smallest_node[2] + k];
//                    val[4] += wOneD[0][i] * wOneD[1][j] * wOneD[2][k] * buffer[4][smallest_node[0] + i][smallest_node[1] + j][smallest_node[2] + k];
//                    val[5] += wOneD[0][i] * wOneD[1][j] * wOneD[2][k] * buffer[5][smallest_node[0] + i][smallest_node[1] + j][smallest_node[2] + k];
//                }
//            }
//        }
//        float local_dt = dt;
//
//        __syncthreads();
//        int parid_mapped = d_indexTrans[parid];
//        d_tmp[3*parid] = (val[3] * (1.0f - flip) + (val[0] + d_sorted_velocities[parid_mapped].x) * flip);
//        d_tmp[3*parid + 1] = (val[4] * (1.0f - flip) + (val[1] + d_sorted_velocities[parid_mapped].y) * flip);
//        d_tmp[3*parid + 2] = (val[5] * (1.0f - flip) + (val[2] + d_sorted_velocities[parid_mapped].z) * flip);
//
//        //vel_m[parid].x = (val[3] * (1.0f - flip) + (val[0] + d_sorted_velocities[parid_mapped].x) * flip);
//        //vel_m[parid].y = (val[4] * (1.0f - flip) + (val[1] + d_sorted_velocities[parid_mapped].y) * flip);
//        //vel_m[parid].z = (val[5] * (1.0f - flip) + (val[2] + d_sorted_velocities[parid_mapped].z) * flip);
//
//        d_sorted_positions[parid].x += val[3] * local_dt;
//        d_sorted_positions[parid].y += val[4] * local_dt;
//        d_sorted_positions[parid].z += val[5] * local_dt;
//
//        for (int i = 0; i < 9; ++i) val[i] = 0.f;
//
//        for (int i = 0; i < 3; ++i) {
//            for (int j = 0; j < 3; ++j) {
//                for (int k = 0; k < 3; ++k) {
//                    // matrix : again column major
//                    val[0] += buffer[3][smallest_node[0] + i][smallest_node[1] + j][smallest_node[2] + k] * wgOneD[0][i] * wOneD[1][j] * wOneD[2][k];
//                    val[1] += buffer[4][smallest_node[0] + i][smallest_node[1] + j][smallest_node[2] + k] * wgOneD[0][i] * wOneD[1][j] * wOneD[2][k];
//                    val[2] += buffer[5][smallest_node[0] + i][smallest_node[1] + j][smallest_node[2] + k] * wgOneD[0][i] * wOneD[1][j] * wOneD[2][k];
//                    val[3] += buffer[3][smallest_node[0] + i][smallest_node[1] + j][smallest_node[2] + k] * wOneD[0][i] * wgOneD[1][j] * wOneD[2][k];
//                    val[4] += buffer[4][smallest_node[0] + i][smallest_node[1] + j][smallest_node[2] + k] * wOneD[0][i] * wgOneD[1][j] * wOneD[2][k];
//                    val[5] += buffer[5][smallest_node[0] + i][smallest_node[1] + j][smallest_node[2] + k] * wOneD[0][i] * wgOneD[1][j] * wOneD[2][k];
//                    val[6] += buffer[3][smallest_node[0] + i][smallest_node[1] + j][smallest_node[2] + k] * wOneD[0][i] * wOneD[1][j] * wgOneD[2][k];
//                    val[7] += buffer[4][smallest_node[0] + i][smallest_node[1] + j][smallest_node[2] + k] * wOneD[0][i] * wOneD[1][j] * wgOneD[2][k];
//                    val[8] += buffer[5][smallest_node[0] + i][smallest_node[1] + j][smallest_node[2] + k] * wOneD[0][i] * wOneD[1][j] * wgOneD[2][k];
//                }
//            }
//        }
//
//        for (int i = 0; i < 9; ++i) val[i] = val[i] * local_dt;
//        val[0] += 1.f; val[4] += 1.f; val[8] += 1.f;
//
//        __syncthreads();
//
//        T F[9], result[9];
//        for (int i = 0; i < 9; ++i) F[i] = d_sorted_F[parid_mapped + i * numParticle];
//
//        matrixMatrixMultiplication(&(val[0]), F, result);
//
//        for (int i = 0; i < 9; ++i) d_tmp[parid + numParticle * (3 + i)] = result[i];
//    }
//}


__global__ void G2P_APIC_CONFLICT_FREE(
    const int numParticle, const int* d_targetPages, const int* d_virtualPageOffsets, const int3* smallest_nodes,
    int* d_block_offsets, int* d_cellids, int* d_indices, int* d_indexTrans, vector3T* d_sorted_positions, vector3T* d_sorted_velocities,
    T** d_channels, T* d_sorted_F, T* d_sorted_B, T* d_tmp, T dt, int** d_adjPage) {
    __shared__ T buffer[688];// [3][8][8][8];

    int pageid = d_targetPages[blockIdx.x] - 1; // from virtual to physical page
    int cellid = d_block_offsets[pageid]; // 
    int relParid = 512 * (blockIdx.x - d_virtualPageOffsets[pageid]) + threadIdx.x;
    int parid = cellid + relParid;

    int block = threadIdx.x & 0x3f;
    int ci = block >> 4;
    int cj = (block & 0xc) >> 2;
    int ck = block & 3;

    block = threadIdx.x >> 6;
    int bi = block >> 2;
    int bj = (block & 2) >> 1;
    int bk = block & 1;

    int page_idx = block ? d_adjPage[block - 1][pageid] : pageid;

    // vel
    for (int v = 0; v < 3; ++v) {
        int aa = bi * 4 + ci, bb = bj * 4 + cj, cc = bk * 4 + ck;
        if (aa < 6 && bb < 6 && cc < 6) {
            int sdid = v * 216 + (aa) * 36 + (bb) * 6 + cc;
            buffer[sdid + CONFLICT_FREE_OFFSET(sdid)] =
                *((T*)((unsigned long long)d_channels[1 + v] + (int)page_idx * 8192) + (ci * 16 + cj * 4 + ck));
        }
    }
    __syncthreads();

    int smallest_node[3];
    if (relParid < d_block_offsets[pageid + 1] - d_block_offsets[pageid]) {
        cellid = d_cellids[parid] - 1;
        // quadratic B spline weights

        T wOneD[3][3], wgOneD[3][3];

        smallest_node[0] = smallest_nodes[cellid].x;
        smallest_node[1] = smallest_nodes[cellid].y;
        smallest_node[2] = smallest_nodes[cellid].z;

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

        for (int v = 0; v < 3; ++v) smallest_node[v] = smallest_node[v] & 0x3;

        T val[9]; for (int i = 0; i < 9; ++i) val[i] = 0.f;

        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                for (int k = 0; k < 3; ++k) {
                    // v_pic
                    int sdid0 = (smallest_node[0] + i) * 36 + (smallest_node[1] + j) * 6 + smallest_node[2] + k;
                    val[0] += wOneD[0][i] * wOneD[1][j] * wOneD[2][k] * buffer[sdid0 + CONFLICT_FREE_OFFSET(sdid0)];
                    sdid0 += 216;
                    val[1] += wOneD[0][i] * wOneD[1][j] * wOneD[2][k] * buffer[sdid0 + CONFLICT_FREE_OFFSET(sdid0)];
                    sdid0 += 216;
                    val[2] += wOneD[0][i] * wOneD[1][j] * wOneD[2][k] * buffer[sdid0 + CONFLICT_FREE_OFFSET(sdid0)];
                }
            }
        }

        __syncthreads();
        /*d_tmp[parid] = val[0];
        d_tmp[parid + numParticle] = val[1];
        d_tmp[parid + numParticle * 2] = val[2];*/

        d_sorted_velocities[parid].x = val[0];
        d_sorted_velocities[parid].y = val[1];
        d_sorted_velocities[parid].z = val[2];

        d_sorted_positions[parid].x += val[0] * dt;
        d_sorted_positions[parid].y += val[1] * dt;
        d_sorted_positions[parid].z += val[2] * dt;

        for (int i = 0; i < 9; ++i) val[i] = 0.f;
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                for (int k = 0; k < 3; ++k) {
                    // F
                    int sdid0 = (smallest_node[0] + i) * 36 + (smallest_node[1] + j) * 6 + smallest_node[2] + k;
                    int sdid1 = sdid0 + 216, sdid2 = sdid1 + 216;
                    val[0] += buffer[sdid0 + CONFLICT_FREE_OFFSET(sdid0)] * wgOneD[0][i] * wOneD[1][j] * wOneD[2][k];
                    val[1] += buffer[sdid1 + CONFLICT_FREE_OFFSET(sdid1)] * wgOneD[0][i] * wOneD[1][j] * wOneD[2][k];
                    val[2] += buffer[sdid2 + CONFLICT_FREE_OFFSET(sdid2)] * wgOneD[0][i] * wOneD[1][j] * wOneD[2][k];
                    val[3] += buffer[sdid0 + CONFLICT_FREE_OFFSET(sdid0)] * wOneD[0][i] * wgOneD[1][j] * wOneD[2][k];
                    val[4] += buffer[sdid1 + CONFLICT_FREE_OFFSET(sdid1)] * wOneD[0][i] * wgOneD[1][j] * wOneD[2][k];
                    val[5] += buffer[sdid2 + CONFLICT_FREE_OFFSET(sdid2)] * wOneD[0][i] * wgOneD[1][j] * wOneD[2][k];
                    val[6] += buffer[sdid0 + CONFLICT_FREE_OFFSET(sdid0)] * wOneD[0][i] * wOneD[1][j] * wgOneD[2][k];
                    val[7] += buffer[sdid1 + CONFLICT_FREE_OFFSET(sdid1)] * wOneD[0][i] * wOneD[1][j] * wgOneD[2][k];
                    val[8] += buffer[sdid2 + CONFLICT_FREE_OFFSET(sdid2)] * wOneD[0][i] * wOneD[1][j] * wgOneD[2][k];
                }
            }
        }

        for (int i = 0; i < 9; ++i) val[i] = val[i] * dt;
        val[0] += 1.f;
        val[4] += 1.f;
        val[8] += 1.f;

        T F[9];
        __syncthreads();
        int parid_trans = d_indexTrans[parid];
        for (int i = 0; i < 9; ++i) F[i] = d_sorted_F[parid_trans + i * numParticle];

        T result[9];
        matrixMatrixMultiplication(&(val[0]), F, result);

        for (int i = 0; i < 9; ++i) d_tmp[parid + (i + 3) * numParticle] = result[i];

        for (int i = 0; i < 9; ++i) val[i] = 0.f;
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                for (int k = 0; k < 3; ++k) {
                    // B
                    int sdid0 = (smallest_node[0] + i) * 36 + (smallest_node[1] + j) * 6 + smallest_node[2] + k;
                    int sdid1 = sdid0 + 216, sdid2 = sdid1 + 216;
                    val[0] += wOneD[0][i] * wOneD[1][j] * wOneD[2][k] * buffer[sdid0 + CONFLICT_FREE_OFFSET(sdid0)] * (i * dx - xp[0]);
                    val[1] += wOneD[0][i] * wOneD[1][j] * wOneD[2][k] * buffer[sdid1 + CONFLICT_FREE_OFFSET(sdid1)] * (i * dx - xp[0]);
                    val[2] += wOneD[0][i] * wOneD[1][j] * wOneD[2][k] * buffer[sdid2 + CONFLICT_FREE_OFFSET(sdid2)] * (i * dx - xp[0]);
                    val[3] += wOneD[0][i] * wOneD[1][j] * wOneD[2][k] * buffer[sdid0 + CONFLICT_FREE_OFFSET(sdid0)] * (j * dx - xp[1]);
                    val[4] += wOneD[0][i] * wOneD[1][j] * wOneD[2][k] * buffer[sdid1 + CONFLICT_FREE_OFFSET(sdid1)] * (j * dx - xp[1]);
                    val[5] += wOneD[0][i] * wOneD[1][j] * wOneD[2][k] * buffer[sdid2 + CONFLICT_FREE_OFFSET(sdid2)] * (j * dx - xp[1]);
                    val[6] += wOneD[0][i] * wOneD[1][j] * wOneD[2][k] * buffer[sdid0 + CONFLICT_FREE_OFFSET(sdid0)] * (k * dx - xp[2]);
                    val[7] += wOneD[0][i] * wOneD[1][j] * wOneD[2][k] * buffer[sdid1 + CONFLICT_FREE_OFFSET(sdid1)] * (k * dx - xp[2]);
                    val[8] += wOneD[0][i] * wOneD[1][j] * wOneD[2][k] * buffer[sdid2 + CONFLICT_FREE_OFFSET(sdid2)] * (k * dx - xp[2]);

                }
            }
        }

        for (int i = 0; i < 9; ++i) d_tmp[parid + (i + 12) * numParticle] = val[i];
    }
}


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
    int** d_adjPage) {
    __shared__ T buffer[8][8][8];

    int pageid = d_targetPages[blockIdx.x] - 1; // from virtual to physical page
    int cellid = d_block_offsets[pageid]; // 
    int relParid = 512 * (blockIdx.x - d_virtualPageOffsets[pageid]) + threadIdx.x;
    int parid = cellid + relParid;

    int block = threadIdx.x & 0x3f;
    int ci = block >> 4;
    int cj = (block & 0xc) >> 2;
    int ck = block & 3;

    block = threadIdx.x >> 6;
    int bi = block >> 2;
    int bj = (block & 2) >> 1;
    int bk = block & 1;

    int page_idx = block ? d_adjPage[block - 1][pageid] : pageid;

    buffer[bi * 4 + ci][bj * 4 + cj][bk * 4 + ck] = d_channels[page_idx * 64 + (ci * 16 + cj * 4 + ck)];
    
    __syncthreads();

    int smallest_node[3];
    if (relParid < d_block_offsets[pageid + 1] - d_block_offsets[pageid]) {
        cellid = d_cellids[parid] - 1;
        // quadratic B spline weights

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

        for (int v = 0; v < 3; ++v) smallest_node[v] = smallest_node[v] & 0x3;

        T val[3]; for (int i = 0; i < 3; i++) val[i] = 0;

        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                for (int k = 0; k < 3; ++k) {

                    T wg[3];
                    wg[0] = wgOneD[0][i] * wOneD[1][j] * wOneD[2][k];
                    wg[1] = wOneD[0][i] * wgOneD[1][j] * wOneD[2][k];
                    wg[2] = wOneD[0][i] * wOneD[1][j] * wgOneD[2][k];

                    T g_c = buffer[smallest_node[0] + i][smallest_node[1] + j][smallest_node[2] + k];

                    val[0] += wg[0] * g_c;
                    val[1] += wg[1] * g_c;
                    val[2] += wg[2] * g_c;
                }
            }
        }

        __syncthreads();

        T PARA = d_sorted_vol[parid_trans] * 4.f * L0 * L0;
        d_sorted_col[parid_trans].x = val[0]* PARA;
        d_sorted_col[parid_trans].y = val[1]* PARA;
        d_sorted_col[parid_trans].z = val[2]* PARA;
    }
}

   
