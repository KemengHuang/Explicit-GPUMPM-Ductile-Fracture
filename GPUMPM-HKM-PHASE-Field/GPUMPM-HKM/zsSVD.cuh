#include <cuda_runtime.h>
struct Matrix3x3d {
	double m[3][3];
};
__device__ 
void mSVD(const Matrix3x3d& F, Matrix3x3d& Uout, Matrix3x3d& Vout, Matrix3x3d& Sigma);