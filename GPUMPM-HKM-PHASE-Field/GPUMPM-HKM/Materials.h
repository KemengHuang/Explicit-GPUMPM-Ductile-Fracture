#pragma once
#ifndef __MATERIALS_H_
#define __MATERIALS_H_
#include "Setting.h"
class Material {
public:
    Material() {};
    int getMaterialType() {
        return _materialType;
    }
protected:
    int _materialType;
};

class ElasticMaterial : public Material {
public:
    ElasticMaterial() = delete;
    ElasticMaterial(int materialType, T ym, T pr, T d, T vol);
    T   _youngsModulus, _poissonRatio, _density;
    T   _lambda, _mu, _volume;
    T   _kappa;
    T parabolic_M;
};
#endif