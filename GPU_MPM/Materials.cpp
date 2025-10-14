#include"Materials.h"


ElasticMaterial::ElasticMaterial(int materialType, T ym, T pr, T d, T vol) {
	_materialType = materialType;
	_youngsModulus = ym;
	_poissonRatio = pr;
	_density = d;
	_volume = vol;
	_lambda = _youngsModulus * _poissonRatio / ((1 + _poissonRatio) * (1 - 2 * _poissonRatio));
	_mu = _youngsModulus / (2 * (1 + _poissonRatio));
	_kappa = _youngsModulus / (3 * (1 - 2 * _poissonRatio));
	parabolic_M = 15.f;
}