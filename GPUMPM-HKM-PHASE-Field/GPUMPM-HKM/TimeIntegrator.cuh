
#ifndef __TIMEINTEGRATOR_H_
#define __TIMEINTEGRATOR_H_
#include "Setting.h"
#include "Models.h"
#include "SPGrid.h"
#include "Transformer.cuh"
#include "math.h"
#include <memory>

class MPMTimeIntegrator {
public:
	MPMTimeIntegrator() {};
	MPMTimeIntegrator(int transferScheme, int numParticle, T* dMemTrunk);
	~MPMTimeIntegrator();

	virtual void integrate(int type, float* time1, float* time2, const T dt, Model& model, std::unique_ptr<SPGrid>& grid, std::unique_ptr<DomainTransformer>& trans) = 0;


	void initMatrix(unsigned int num_particle, T* matrix);
	void computeCellIndex(std::unique_ptr<DomainTransformer>& trans, std::unique_ptr<Particles>& particle);
	void transferP2G(const T dt, std::unique_ptr<Particles>& particles, std::unique_ptr<SPGrid>& grid, std::unique_ptr<DomainTransformer>& trans);

	//void transferVolP2G(const T dt, Model& model, std::unique_ptr<SPGrid>& grid, std::unique_ptr<DomainTransformer>& trans);
	//void preCondition(const T dt, Model& model, std::unique_ptr<SPGrid>& grid, std::unique_ptr<DomainTransformer>& trans);
	//	void call_postP2G(std::unique_ptr<SPGrid>& grid, std::unique_ptr<DomainTransformer>& trans);
	void undateGrid(const T dt, std::unique_ptr<SPGrid>& grid, std::unique_ptr<DomainTransformer>& trans);

	//void transGrid(std::unique_ptr<SPGrid>& grid, std::unique_ptr<DomainTransformer>& trans, int input);

	void PullGrid(std::unique_ptr<SPGrid>& grid, std::unique_ptr<DomainTransformer>& trans, T dt);

	void resolveCollision(std::unique_ptr<SPGrid>& grid, std::unique_ptr<DomainTransformer>& trans);
	//	void call_preG2P(std::unique_ptr<SPGrid>& grid, std::unique_ptr<DomainTransformer>& trans);
	void transferG2P(const T dt, std::unique_ptr<Particles>& geometry, std::unique_ptr<SPGrid>& grid, std::unique_ptr<DomainTransformer>& trans);



protected:
	int   _transferScheme;
	int _numParticle;
	T* d_contribution;
	T* d_memTrunk;
	//T* d_tmp;
};

class ExplicitTimeIntegrator : public MPMTimeIntegrator {
	friend class MPMSimulator;
public:
	ExplicitTimeIntegrator(int transferScheme, int numParticle, T* dMemTrunk);
	~ExplicitTimeIntegrator() {};
	virtual void integrate(int type, float* time1, float* time2,
		const T dt,
		Model& model,
		std::unique_ptr<SPGrid>& grid,
		std::unique_ptr<DomainTransformer>& trans);

	void computeForceCoefficient(Model& model);
	void computeParticlePhase_vol(Model& model);
	//void updateGridVelocity(const T dt, std::unique_ptr<SPGrid>& grid, std::unique_ptr<DomainTransformer>& trans);
};

#endif