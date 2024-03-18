#include"LoadSample.h"

void Sample::LoadSDF(std::string filename, T& pDx, T& minx, T& miny, T& minz, int& ni, int& nj, int& nk)
{
	T	m_dx;
	T m_minBox[3];
	std::ifstream infile(filename);

	

	infile >> m_ni >> m_nj >> m_nk;
	infile >> m_minBox[0] >> m_minBox[1] >> m_minBox[2];
	infile >> m_dx;

	//std::cout << m_dx << std::endl;

	pDx = m_dx;

	ni = m_ni;
	nj = m_nj;
	nk = m_nk;
	minx = m_minBox[0];
	miny = m_minBox[1];
	minz = m_minBox[2];

	std::cout << "load grid size " << m_ni << ", " << m_nj << ", " << m_nk << std::endl;

	int gridSize = m_ni * m_nj * m_nk;
	m_phiGrid.resize(gridSize);
	for (int i = 0; i < gridSize; ++i) {
		infile >> m_phiGrid[i];
	}
	infile.close();
}

inline T Sample::fetchGrid(int i, int j, int k) 
{ 
	return m_phiGrid[(i + m_ni * (j + m_nj * k))]; 
}

inline T Sample::fetchGridTrilinear(T x, T y, T z)
{
	T dx = x - floor(x);
	T dy = y - floor(y);
	T dz = z - floor(z);
	
	T c000 = fetchGrid((int)floor(x), (int)floor(y), (int)floor(z));
	T c001 = fetchGrid((int)floor(x), (int)floor(y), (int)ceil(z));
	T c010 = fetchGrid((int)floor(x), (int)ceil(y), (int)floor(z));
	T c011 = fetchGrid((int)floor(x), (int)ceil(y), (int)ceil(z));
	T c100 = fetchGrid((int)ceil(x), (int)floor(y), (int)floor(z));
	T c101 = fetchGrid((int)ceil(x), (int)floor(y), (int)ceil(z));
	T c110 = fetchGrid((int)ceil(x), (int)ceil(y), (int)floor(z));
	T c111 = fetchGrid((int)ceil(x), (int)ceil(y), (int)ceil(z));
	T c00 = c000 * (1 - dx) + c100 * dx;
	T c01 = c001 * (1 - dx) + c101 * dx;
	T c10 = c010 * (1 - dx) + c110 * dx;
	T c11 = c011 * (1 - dx) + c111 * dx;
	T c0 = c00 * (1 - dy) + c10 * dy;
	T c1 = c01 * (1 - dy) + c11 * dy;
	return c0 * (1 - dz) + c1 * dz;
}

int Sample::GenerateUniformSamples(T samplesPerCell, std::vector<T>& outputSamples)
{
	// get total sample number
	int validCellNum = 0;
	for (int i = 0; i < m_ni - 1; i++)
	{
		for (int j = 0; j < m_nj - 1; j++)
		{
			for (int k = 0; k < m_nk - 1; k++)
			{
				if (fetchGrid(i, j, k) < 0 ||
					fetchGrid(i, j, k + 1) < 0 ||
					fetchGrid(i, j + 1, k) < 0 ||
					fetchGrid(i, j + 1, k + 1) < 0 ||
					fetchGrid(i + 1, j, k) < 0 ||
					fetchGrid(i + 1, j, k + 1) < 0 ||
					fetchGrid(i + 1, j + 1, k) < 0 ||
					fetchGrid(i + 1, j + 1, k + 1) < 0)
					validCellNum++;
			}
		}
	}

	int sampleNum = validCellNum * samplesPerCell;

	for (int i = 0; i < sampleNum; i++)
	{
		T tmpPoint[3];
		do
		{
			tmpPoint[0] = (T)rand() / RAND_MAX * (m_ni - 1);
			tmpPoint[1] = (T)rand() / RAND_MAX * (m_nj - 1);
			tmpPoint[2] = (T)rand() / RAND_MAX * (m_nk - 1);
		} while (fetchGridTrilinear(tmpPoint[0], tmpPoint[1], tmpPoint[2]) >= 0);

		outputSamples.push_back(tmpPoint[0]);
		outputSamples.push_back(tmpPoint[1]);
		outputSamples.push_back(tmpPoint[2]);
	}

	return sampleNum;
}



std::vector<vector3T> Sample::Initial_data(unsigned int* center, unsigned int* res, std::string fileName) {
	std::vector<vector3T> data;
	int minCorner[3];
	for (int i = 0; i < 3; ++i)
		minCorner[i] = center[i] - .5 * res[i];

	minCorner[1] = center[1] + 0.25 * res[1];

	
	int samplePerCell = 20;
	int offsetx = minCorner[0];
	int offsety = minCorner[1];
	int offsetz = minCorner[2];
	int width = res[0];
	int height = res[1];
	int depth = res[2];

	T levelsetDx;
	std::vector<T>  samples;

	T levesetMinx, levelsetMiny, levelsetMinz;
	int levelsetNi, levelsetNj, levelsetNk;

	LoadSDF(fileName, levelsetDx, levesetMinx, levelsetMiny, levelsetMinz, levelsetNi, levelsetNj, levelsetNk);

	int minx = 1, miny = 1, minz = 1;
	int maxx = levelsetNi - 2, maxy = levelsetNj - 2, maxz = levelsetNk - 2;

	T   scalex = 1.f * width / (maxx - minx);
	T   scaley = 1.f * height / (maxy - miny);
	T   scalez = 1.f * depth / (maxz - minz);

	T   scale = scalex < scaley ? scalex : scaley;
	scale = scalez < scale ? scalez : scale;

	T samplePerLevelsetCell = samplePerCell * scale * scale * scale;

	GenerateUniformSamples(samplePerLevelsetCell, samples);

	for (int i = 0, size = samples.size() / 3; i < size; i++) {
		vector3T pcle;
		pcle.x = ((samples[i * 3 + 0] - minx) * scale + offsetx + 4) * dx;
		pcle.y = ((samples[i * 3 + 1] - miny) * scale + offsety + 4) * dx;
		pcle.z = ((samples[i * 3 + 2] - minz) * scale + offsetz + 4) * dx;
		data.push_back(pcle);
		/*vector3T pcle2;
		pcle2.x = ((samples[i * 3 + 0] - minx) * scale + offsetx + 4) * dx;
		pcle2.y = ((samples[i * 3 + 1] - miny) * scale + offsety + 4) * dx-0.3f;
		pcle2.z = ((samples[i * 3 + 2] - minz) * scale + offsetz + 4) * dx;
		data.push_back(pcle2);*/
	}
	return data;
}

std::vector<vector3T> Sample::readObj(std::string filename) {   ///< cube
	std::vector<vector3T> data;

	std::ifstream in(filename);
	char v;
	T x, y, z;
	while (in >> v >> x >> y >> z) {
		vector3T vertex;
		vertex.x = x; vertex.y = y-0.2f; vertex.z = z;
		data.push_back(vertex);
	}
	in.close();
	return data;
}


std::vector<vector3T> Sample::Initialize_Data_cube() {   ///< cube
	std::vector<vector3T> data;

	int Xmin = N / 2 - N / 20;
	int Xmax = N / 2 + N / 20;

	int Ymin = N / 2 - N / 20;
	int Ymax = N / 2 + N / 20;

	int Zmin = N / 2 - N / 8;
	int Zmax = N / 2 + N / 8;
	//T dx = (T)1. / (T)N;
	for (int i = Xmin; i < Xmax; ++i)
		for (int j = Ymin; j < Ymax; ++j)
			for (int k = Zmin; k < Zmax; ++k)
			{
				T cell_center[3];
				cell_center[0] = (i + 0.5) * dx;
				cell_center[1] = (j + 0.5) * dx;
				cell_center[2] = (k + 0.5) * dx;
				int numberPc = 1;
				T ddx = dx / (2 * numberPc + 1);
				for (int ii = -numberPc; ii <= numberPc; ii = ii + 1)
					for (int jj = -numberPc; jj <= numberPc; jj = jj + 1)
						for (int kk = -numberPc; kk <= numberPc; kk = kk + 1)
						{
							vector3T particle;
							//std::array<T, Dim> particle;
							particle.x = cell_center[0] + ii * ddx;
							particle.y = cell_center[1] + jj * ddx;
							particle.z = cell_center[2] + kk * ddx;
							data.push_back(particle);
						}
			}
	return data;
}