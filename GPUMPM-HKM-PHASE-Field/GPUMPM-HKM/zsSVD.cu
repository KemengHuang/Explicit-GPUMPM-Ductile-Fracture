#include "zsSVD.cuh"
#include "zensim/math/matrix/QRSVD.hpp"
#include "zensim/math/bit/Bits.h"
#include "math.h"

__device__ void mQRSVD(const Matrix3x3d& F, Matrix3x3d& Uout, Matrix3x3d& Vout, Matrix3x3d& Sigma) {

	using matview = zs::vec_view<double, zs::integer_seq<int, 3, 3>>;
	using cmatview = zs::vec_view<const double, zs::integer_seq<int, 3, 3>>;
	using vec3 = zs::vec<double, 3>;
	cmatview F_{ (const double*)F.m };
	matview UU{ (double*)Uout.m }, VV{ (double*)Vout.m };
	vec3 SS{};
	zs::tie(UU, SS, VV) = zs::math::qr_svd(F_);
	for (int i = 0; i != 3; ++i)
		for (int j = 0; j != 3; ++j) {
			Uout.m[i][j] = UU(i, j);
			Vout.m[i][j] = VV(i, j);
			Sigma.m[i][j] = (i != j ? 0. : SS[i]);
		}
}


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
) {
	Matrix3x3d F, U, V, S;
	F.m[0][0] = a11;
	F.m[0][1] = a12;
	F.m[0][2] = a13;
	F.m[1][0] = a21;
	F.m[1][1] = a22;
	F.m[1][2] = a23;
	F.m[2][0] = a31;
	F.m[2][1] = a32;
	F.m[2][2] = a33;

	mQRSVD(F, U, V, S);

	u11 = U.m[0][0];
	u12 = U.m[0][1];
	u13 = U.m[0][2];
	u21 = U.m[1][0];
	u22 = U.m[1][1];
	u23 = U.m[1][2];
	u31 = U.m[2][0];
	u32 = U.m[2][1];
	u33 = U.m[2][2];

	v11 = V.m[0][0];
	v12 = V.m[0][1];
	v13 = V.m[0][2];
	v21 = V.m[1][0];
	v22 = V.m[1][1];
	v23 = V.m[1][2];
	v31 = V.m[2][0];
	v32 = V.m[2][1];
	v33 = V.m[2][2];

	s11 = S.m[0][0];
	s22 = S.m[1][1];
	s33 = S.m[2][2];
}