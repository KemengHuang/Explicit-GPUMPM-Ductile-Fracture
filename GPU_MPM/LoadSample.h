#pragma once

#include<vector>
#include<array>
#include"Setting.h"
#include<string>
#include<fstream>
#include<iostream>

class Sample {
public:
	Sample() {};
	~Sample() {}
	void LoadSDF(std::string filename, T& pDx, T& minx, T& miny, T& minz, int& ni, int& nj, int& nk);
	inline T fetchGrid(int i, int j, int k);// { return m_phiGrid[i + m_ni * (j + m_nj * k)]; }
	int GenerateUniformSamples(T samplesPerCell, std::vector<T>& outputSamples);
	inline T fetchGridTrilinear(T x, T y, T z);
	std::vector<vector3T> Initial_data(unsigned int* center, unsigned int* res, std::string fileName);
	std::vector<vector3T> readObj(std::string fileName);
	std::vector<vector3T> Initialize_Data_cube();
private:
	int m_ni, m_nj, m_nk;
	std::vector<T> m_phiGrid;
};






