#include "zsSVD.cuh"
#include "zensim/math/matrix/QRSVD.hpp"
#include "zensim/math/bit/Bits.h"
#include "math.h"

__device__ void mSVD(const Matrix3x3d& F, Matrix3x3d& Uout, Matrix3x3d& Vout, Matrix3x3d& Sigma) {

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