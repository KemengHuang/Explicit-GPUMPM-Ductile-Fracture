#include <cuda_runtime.h>
struct Matrix3x3d {
	double m[3][3];
};
__device__ 
void mQRSVD(const Matrix3x3d& F, Matrix3x3d& Uout, Matrix3x3d& Vout, Matrix3x3d& Sigma);

__device__
void msvd(
	double a11, double a12, double a13, double a21, double a22, double a23, double a31, double a32, double a33,			// input A     
	double& u11, double& u12, double& u13, double& u21, double& u22, double& u23, double& u31, double& u32, double& u33,	// output U      
	double& s11,
	//float &s12, float &s13, float &s21, 
	double& s22,
	//float &s23, float &s31, float &s32, 
	double& s33,	// output S
	double& v11, double& v12, double& v13, double& v21, double& v22, double& v23, double& v31, double& v32, double& v33	// output V
);