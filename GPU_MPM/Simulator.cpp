#include "Simulator.h"
#include"SPGridMask.h"
#include"LoadSample.h"
#include"cuda_tools.h"
#include"cuda_kernal_tools.cuh"
//#include "partio.h"
void matrixVectorMultiplication(const T* x, const T* v, vector3T *result)
{
    result->x = x[0] * v[0] + x[3] * v[1] + x[6] * v[2];
    result->y = x[1] * v[0] + x[4] * v[1] + x[7] * v[2];
    result->z = x[2] * v[0] + x[5] * v[1] + x[8] * v[2];
}

MPMSimulator::~MPMSimulator() { FreeDeviceMEM(); }

void MPMSimulator::MallocDeviceMEM() {
	int ts = tableSize << 6;
	int maxSize = ts > numParticle ? ts : numParticle;
	maxSize = maxSize > space_page_num ? maxSize : space_page_num;
	Allocate_Prefix_Sum_RecursiveMem_Int(maxSize);
	//CUDA_SAFE_CALL(cudaMalloc((void**)&recursiveSum, maxSize * sizeof(int)));

    CUDA_SAFE_CALL(cudaMalloc((void**)&d_memTrunk, sizeof(T) * numParticle * (18))); // tmp(21)         
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_maxVelSquared, sizeof(T)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_masks, sizeof(unsigned long long) * Dim));
    CUDA_SAFE_CALL(cudaMemcpy(d_masks, h_masks, sizeof(unsigned long long) * Dim, cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_keyTable, sizeof(unsigned long long) * tableSize));
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_valTable, sizeof(int) * tableSize));
    //CUDA_SAFE_CALL(cudaMalloc((void**)&d_pageId, sizeof(int) * space_page_num));
	//CUDA_SAFE_CALL(cudaMalloc((void**)&d_page_offset, sizeof(int) * (space_page_num+1)));
}

void MPMSimulator::FreeDeviceMEM() {
    CUDA_SAFE_CALL(cudaFree(d_memTrunk)); // tmp(21)         
    CUDA_SAFE_CALL(cudaFree(d_maxVelSquared));
    CUDA_SAFE_CALL(cudaFree(d_masks));
    CUDA_SAFE_CALL(cudaFree(d_keyTable));
    CUDA_SAFE_CALL(cudaFree(d_valTable));
    //CUDA_SAFE_CALL(cudaFree(d_pageId));
	//CUDA_SAFE_CALL(cudaFree(d_page_offset));
	Free_Prefix_Sum_RecursiveMem_Int();
}


void MPMSimulator::buildModel(const int materialType, const T youngsModulus,
    const T poissonRatio, const T density, const T volume, int idx) {
    auto particleGroup = std::make_unique<Particles>();

    particleGroup->initialize(numParticle, tableSize, h_mass.data(), (h_pos).data(), (h_vel).data(), d_masks, 
        d_keyTable, d_valTable, d_memTrunk, d_pageId,d_page_offset);

    auto material = std::make_unique<ElasticMaterial>(materialType, youngsModulus, poissonRatio, density, volume);
    h_models.emplace_back(std::make_unique<Model>(std::move(particleGroup), std::move(material)));
}

void MPMSimulator::buildGrid(const int width, const int height, const int depth, const T dc) {    
    h_pGrid = std::make_unique<SPGrid>();
    h_pGrid->initialize(d_masks, h_masks, width, height, depth, dc);
    h_pGrid->SPGrid_MEM_Malloc();
}
void MPMSimulator::buildTransformer(const int gridVolume) {
    h_pTrans = std::make_unique<DomainTransformer>();
    h_pTrans->initialize(numParticle, gridVolume, d_masks, h_masks, getModelParticlePtr(0)->cell_id, tableSize, d_keyTable, d_valTable);
    h_pTrans->Transformer_Malloc_MEM();
}

void MPMSimulator::buildIntegrator(const int integratorType, const int transferScheme, const T dtDefault) {
    h_pTimeIntegrator = std::make_unique<ExplicitTimeIntegrator>(transferScheme, numParticle, d_memTrunk);
    _dtDefault = dtDefault;
    h_pTimeIntegrator->initMatrix(numParticle, getModelParticlePtr()->d_F);
    if (transferScheme)
        CUDA_SAFE_CALL(cudaMemset(getModelParticlePtr()->d_B, 0, sizeof(T) * 9 * numParticle));
}


bool MPMSimulator::build(unsigned int buildtype) {    
    SPGridMask spgdm(Log2(sizeof(CH_STRUCT)), Log2(sizeof(T)));
    spgdm.Cal_Masks();
    spgdm.getMasks(h_masks);

    T totalMass, volume, pMass;
    unsigned int integratorType, transferScheme;
    T dtDefault;

    unsigned int materialType = 1;   // 0 neohookean 1 fixed corotated
    T youngsModulus = 2000.f;
    T poissonRatio = 0.4;
    T density = rateSize*rateSize*rateSize*3.f;
    T totalVolume = 1.f;
    //auto assets_dir = std::string{ gipc::assets_dir() };
    auto assets_path = std::string{ gmpm_ASSETS_DIR };
    if (buildtype == 0) {
        int geometryType = 0;
        //tableSize = 100000;
        tableSize = 36000 * rateSize * rateSize * rateSize * MEMORY_SCALE;

        Sample sple;
        h_pos = sple.readObj(assets_path + "sample/tube_100.obj");//sple.Initialize_Data_cube();
        //h_pos = sple.Initialize_Data_cube();
        numParticle = h_pos.size();
        totalVolume = 0.75f * 0.75f * 0.75f;
        // velocity
        h_vel.resize(numParticle);
        std::fill(h_vel.begin(), h_vel.end(), vector3T{ 0, 0, 0 });

        // integrator
        integratorType = MPM_SIM_TYPE;		// 0 explicit 1 implicit
        transferScheme = TRANSFER_SCHEME;	// 0 flip 1 apic 2 mls
        dtDefault = integratorType == 0 ? 1e-4 : 1e-3;

    }
    else if (buildtype == 1) {
        int geometryType = 7;
		tableSize = 36000 * rateSize * rateSize * rateSize * MEMORY_SCALE;
        srand(0);
        // position
        unsigned int center[3] = { N / 2,N / 2,N / 2 };
        unsigned int res[3] = { 5 * N / 6, 5 * N / 6, 5 * N / 6 };
        unsigned int minCorner[3];
        for (int i = 0; i < 3; ++i)
            minCorner[i] = center[i] - 0.5 * res[i];

        Sample sple;
        std::string fileName = assets_path+"sample/two_dragons.sdf";
        h_pos = sple.Initial_data(center, res, fileName);


        int sizeOfOneCopy = h_pos.size();
        T stride = (N - center[0] * 2) * 0.3333 * dx;

        for (int i = 0, j = 0; j < 0; ++j){
            if (i == 0 && j == 0) continue;
            T theta = (T)(10. / 180.) * 3.1415926f;
            T rotation[9] = {
                std::cos(theta),-std::sin(theta),0,
                std::sin(theta),std::cos(theta),0,
                0,0,1
            };

            for (int p = 0; p < sizeOfOneCopy; ++p){
                vector3T pos = h_pos[p];
                // move pos to center 
                T diff[3];
                diff[0] = pos.x - center[0] * dx;
                diff[1] = pos.y - center[1] * dx;
                diff[2] = pos.z - center[2] * dx;
                matrixVectorMultiplication(rotation, diff, &pos);
                pos.x = pos.x + center[0] * dx;
                pos.y = pos.y + center[1] * dx;
                pos.z = pos.z + center[2] * dx;
            }
        }

        totalVolume = res[0] * res[1] * res[2] * dx * dx * dx;

        numParticle = h_pos.size();


        // velocity
        for (int i = 0; i < numParticle; i++) {
            vector3T vel = { 0.f,0.f,0.f };
            h_vel.push_back(vel);
        }
        int numPartvector3T = 0;
        for (int i = 0; i < numParticle; i++){
            if (h_pos[i].z > 0.5f) {
                numPartvector3T++;
            }
        }
              
        std::cout << "The Number of vector3Ts for the first part" << numPartvector3T << std::endl;
        
        std::vector<vector3T> tmp = h_pos;
        int cubeIndex = 0;
        int baseIndex = numPartvector3T;
        for (int i = 0; i < numParticle; i++) {
            if (tmp[i].z > 0.5) {
                h_pos[cubeIndex] = tmp[i];
                h_vel[cubeIndex++].z = -0.5;
            }
            else {
                h_pos[baseIndex] = tmp[i];
                h_vel[baseIndex++].z = +0.5;
            }
        }
        tmp.clear();
        
        // integrator
        integratorType = MPM_SIM_TYPE;		// 0 explicit 1 implicit
        transferScheme = TRANSFER_SCHEME;	// 0 flip 1 apic 2 mls
        dtDefault = (integratorType == 0 ? 1e-4 : 1e-3);
    }
    // file
    h_indices.resize(numParticle);
    for (int i = 0; i < numParticle; ++i) h_indices[i] = i;
    int numPartParticle = 0;
    totalMass = density * totalVolume;
    volume = totalVolume / numParticle;
    pMass = totalMass / numParticle;
    // mass
    h_mass.resize(numParticle);
    for (auto& mass : h_mass) mass = pMass;


    MallocDeviceMEM();
    


    buildModel(materialType, youngsModulus, poissonRatio, density, volume, 0);
    buildGrid(N, N, N, dx);


    buildTransformer(N * N * N);
    buildIntegrator(integratorType, transferScheme, dtDefault);
    //FreeDeviceMEM();
    return true;
}

T MPMSimulator::computeDt(const T cur, const T next) {
    T dt = _dtDefault, maxVel = 0;

    //CUDA_SAFE_CALL(cudaMemcpy(d_maxVelSquared, (void*)&maxVel, sizeof(T), cudaMemcpyHostToDevice));
    getModelParticlePtr()->getMaxParticleVel_c(&maxVel);
	//getModelParticlePtr()->getMaxParticleVel(d_maxVelSquared);
    //CUDA_SAFE_CALL(cudaMemcpy((void*)&maxVel, d_maxVelSquared, sizeof(T), cudaMemcpyDeviceToHost));
    //printf("maxVel   %f\n", maxVel);
    maxVel = sqrt(maxVel);
    //printf("maxVel is %f\n ", maxVel);

	//printf("current maxVel %f\n", maxVel);

    if (maxVel > 0 && (maxVel = dx * .3f / maxVel) < _dtDefault) dt = maxVel;


	//printf("current dt %f\n", dt);
	
    //if (cur + dt >= next)
    //    dt = next - cur;
    //else if (cur + 2 * dt >= next && (maxVel = (next - cur) * 0.51) < dt)
    //    dt = maxVel;

    //printf(" dt is %f curr is %f and next is %f \n", dt, cur, next);
    return dt;
}
int substep = 0;
void MPMSimulator::simulateStick(float* cflTime, float* preTime, float* simuTime, int type)
{
	cudaEvent_t start, end;
	cudaEventCreate(&start);
	cudaEventCreate(&end);

	T timeToF = _currentFrame / _frameRate;

	//printf("          current substep %d\n", substep);

	cudaEventRecord(start);
	T dt = computeDt(_currentTime, timeToF);

	cudaEventRecord(end);
	CUDA_SAFE_CALL(cudaDeviceSynchronize());
	cudaEventElapsedTime(cflTime, start, end);


	(cudaEventDestroy(start));
	(cudaEventDestroy(end));
	substep++;

	if (dt > 0.f) {
		getTimeIntegratorPtr()->integrate(type, simuTime, preTime, dt, getModel(0), getGridPtr(), getTransformerPtr());
	}

	_currentFrame++;
	getModelParticlePtr()->Attribs_Back_Host(h_pos, h_vel, h_indices);
	bool output = false;
	if (output) {

	}
}

