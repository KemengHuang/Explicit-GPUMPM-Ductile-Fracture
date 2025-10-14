#pragma once
#include"Setting.h"
#include"SPGrid.h"
#include "Models.h"
#include"Transformer.cuh"
#include"TimeIntegrator.cuh"
#include<vector>
#include<string>



class MPMSimulator {
public:
	MPMSimulator() { _currentFrame = 0; _currentFrame = 0.f; _frameRate = 48.f; }
	~MPMSimulator();
	bool build(unsigned int buildtype);
	void MallocDeviceMEM();
	void FreeDeviceMEM();
	void buildModel(const int materialType, const T youngsModulus, 
					const T poissonRatio, const T density, const T volume, int idx = 0);
	void buildGrid(const int width, const int height, const int depth, const T dc);
	void buildTransformer(const int gridVolume);
	void buildIntegrator(const int integratorType, const int transferScheme, const T dtDefault);

	auto& getTimeIntegratorPtr() { return h_pTimeIntegrator; }
	auto& getGridPtr() { return h_pGrid; }
	auto& getTransformerPtr() { return h_pTrans; }
	Model& getModel(const unsigned int modelId=0) { return h_models[modelId]; }
	auto& getModelParticlePtr(const unsigned int modelId=0) { return getModel(modelId).getParticlesPtr(); }
	auto& getModelMaterialPtr(const unsigned int modelId=0) { return getModel(modelId).getMaterialPtr(); }


	T computeDt(const T cur, const T next);
	void simulateStick(float* cflTime, float* preTime, float* simuTime, int type);
	void writePartio(const std::string& filename);


	std::vector<vector3T>	h_pos;
	unsigned int	numParticle;
private:
	unsigned int			_currentFrame;
	T	        _currentTime;
	T	 _frameRate;
	//MPM
	std::unique_ptr<SPGrid>         h_pGrid;
	std::unique_ptr<DomainTransformer>  h_pTrans;
	std::unique_ptr<MPMTimeIntegrator>  h_pTimeIntegrator;
	std::vector<Model> h_models;
	//HOST
	unsigned long long h_masks[3];
	unsigned int	tableSize;
	
	std::vector<T>	h_mass;
	std::vector<vector3T>	h_vel;
	std::vector<unsigned int>	h_indices;

	//DEVICE
	T*	d_maxVelSquared;
	unsigned long long*	d_masks;
	unsigned long long*	d_keyTable;  
	int*	d_valTable;  
	int*	d_pageId;
	int* d_page_offset;

	T*	d_memTrunk;
	T _dtDefault;

	//int* recursiveSum;
};
