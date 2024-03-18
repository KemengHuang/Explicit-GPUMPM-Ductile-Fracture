#pragma once
#ifndef __MODELS_CUH_
#define __MODELS_CUH_
#include "Setting.h"
#include <memory>
#include "Materials.h"
#include "Particles.cuh"

class Model {
public:
    
    Model() {};
    

    auto& getParticlesPtr() {
        return _particle;
    }
    auto& getMaterialPtr() {
        return _material;
    }

    Model::Model(std::unique_ptr<Model> model) :
        _particle(std::move(model->getParticlesPtr())), _material(std::move(model->getMaterialPtr())) {}

    Model::Model(std::unique_ptr<Particles> particle, std::unique_ptr<Material> material) :
        _particle(std::move(particle)), _material(std::move(material)) {}

private:
    std::unique_ptr<Particles>           _particle;
    std::unique_ptr<Material>   _material;
};

#endif