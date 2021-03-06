/*
 * init.C
 *
 *  Created on: Nov 2, 2012
 *      Author: jianqiao
 */

#include "types.h"
#include "stages.h"
#include "utils.h"
#include "math.h"
#include "string.h"
#include <vector>

SphereArray* initSphereArray(int xdim, int ydim, int zdim,
							 float radius, float dist) {
	std::vector<Sphere> vecSphere;
	for (int i = 0; i < xdim; i++) {
		float x = i * dist;
		for (int j = 0; j < ydim; j++) {
			float y = j * dist;
			for (int k = 0; k < zdim; k++) {
				float z = k * dist;

				Sphere sphere;
				sphere.x = x;
				sphere.y = y;
				sphere.z = z;
				sphere.r = radius;

				vecSphere.push_back(sphere);
			}
		}
	}

	uint numOfSpheres = vecSphere.size();
	Sphere *spheres = new Sphere[numOfSpheres];
	std::copy(vecSphere.begin(), vecSphere.end(), spheres);

	SphereArray *sphereArray = new SphereArray;
	sphereArray->size = numOfSpheres;
	sphereArray->objects = spheres;

	return sphereArray;
}

struct Transformation {
	double mat[4][4];

	Transformation() {
		for (int i = 0; i < 4; i++) {
			for (int j = 0; j < 4; j++) {
				mat[i][j] = 0;
			}
			mat[i][i] = 1;
		}
	}

	static Transformation fromRotate(char axis, double th) {
		Transformation ret;
		if (axis == 'x') {
			double mat[4][4] = { { 1,			0,			0,			0 },
								 { 0,			cos(th),	sin(th),	0 },
								 { 0,			-sin(th),	cos(th),	0 },
								 { 0,			0,			0,			1 } };
			memcpy(ret.mat, mat, sizeof mat);
		}
		else if (axis == 'y') {
			double mat[4][4] = { {	cos(th),	0,			sin(th),	0 },
								 {	0,			1,			0,			0 },
								 {	-sin(th),	0,			cos(th),	0 },
								 {	0,			0,			0,			1 } };
			memcpy(ret.mat, mat, sizeof mat);
		}
		else if (axis == 'z') {
			double mat[4][4] = { {	cos(th),	sin(th),	0,			0 },
								 {	-sin(th),	cos(th),	0,			0 },
								 {	0,			0,			1,			0 },
								 {	0,			0,			0,			1 } };
			memcpy(ret.mat, mat, sizeof mat);
		}

		return ret;
	}

	void combine(Transformation r) {
		double result[4][4];
		for (int i = 0; i < 4; i++) {
			for (int j = 0; j < 4; j++) {
				double sum = 0;
				for (int k = 0; k < 4; k++) {
					sum += r.mat[i][k] * mat[k][j];
				}
				result[i][j] = sum;
			}
		}
		memcpy(mat, result, sizeof result);
	}

	void transform(Sphere &sphere) {
		double w[4] = { sphere.x, sphere.y, sphere.z, 1 };
		double v[4];
		for (int i = 0; i < 4; i++) {
			double sum = 0;
			for (int j = 0; j < 4; j++) {
				sum += mat[i][j] * w[j];
			}
			v[i] = sum;
		}
		sphere.x = v[0];
		sphere.y = v[1];
		sphere.z = v[2];
	}


};

void rotate(SphereArray *sphereArray) {
	Transformation t;
	double PI = 3.14159265358979323846;
	t.combine(Transformation::fromRotate('x', 45 * PI / 180));
	t.combine(Transformation::fromRotate('y', 45 * PI / 180));
//	t.combine(Transformation::fromRotate('z', 45 * PI / 180));

	for (uint i = 0; i < sphereArray->size; i++) {
		t.transform(sphereArray->objects[i]);
	}
}

BinSpherePairArray* stage1To3(SphereArray* sphereArray,
						      float binSize) {
	struct BinSphereBound {
		int minIDX, minIDY, minIDZ;
		int maxIDX, maxIDY, maxIDZ;
	};

	float binRate = 1.0 / binSize;

	uint numOfSpheres = sphereArray->size;
	uint numOfPairs = 0;
	BinSphereBound *bounds = new BinSphereBound[numOfSpheres];
	for (int i = 0; i < numOfSpheres; i++) {
		Sphere &sphere = sphereArray->objects[i];
		BinSphereBound &bound = bounds[i];

		float idx = sphere.x * binRate;
		float idy = sphere.y * binRate;
		float idz = sphere.z * binRate;
		float idr = sphere.r * binRate;

		register float v;
		#define MC_FloorCast(exp) \
			(v = exp, v >= 0) ? static_cast<int>(v) : static_cast<int>(v) - 1;

		bound.minIDX = MC_FloorCast(idx - idr);
		bound.minIDY = MC_FloorCast(idy - idr);
		bound.minIDZ = MC_FloorCast(idz - idr);

		bound.maxIDX = MC_FloorCast(idx + idr);
		bound.maxIDY = MC_FloorCast(idy + idr);
		bound.maxIDZ = MC_FloorCast(idz + idr);

		numOfPairs += (bound.maxIDX - bound.minIDX + 1) *
					  (bound.maxIDY - bound.minIDY + 1) *
					  (bound.maxIDZ - bound.minIDZ + 1);
	}

	BinSpherePair *pairs = new BinSpherePair[numOfPairs];
	int cnt = 0;
	for (uint i = 0; i < numOfSpheres; i++) {
		BinSphereBound &bound = bounds[i];
		for (int idx = bound.minIDX; idx <= bound.maxIDX; idx++) {
			for (int idy = bound.minIDY; idy <= bound.maxIDY; idy++) {
				for (int idz = bound.minIDZ; idz <= bound.maxIDZ; idz++) {
					uint key = MC_Key(idx, idy, idz);
					pairs[cnt].objectID	= i;
					pairs[cnt].binID	= key;
					cnt++;

//					printf("[%u] %d, %d, %d, key = %u\n", i, idx, idy, idz, key);
				}
			}
		}
	}
	delete[] bounds;

	BinSpherePairArray *pairArray = new BinSpherePairArray;
	pairArray->size = numOfPairs;
	pairArray->objects = pairs;

	return pairArray;
}

void stage4(BinSpherePairArray* pairArray) {
	uint numOfPairs = pairArray->size;
	BinSpherePair *arr = pairArray->objects;
	BinSpherePair *auxArr = new BinSpherePair[numOfPairs];
	MC_RadixSort_32_16(arr, binID, auxArr, numOfPairs);
	delete[] auxArr;

//	for (uint i = 0; i < numOfPairs; i++) \
		printf("[%u] %u %u\n", i, arr[i].binID, arr[i].objectID);
}

BinSphereDataArray *stage5To6(BinSpherePairArray *pairArray) {
	BinSphereData *data = new BinSphereData[pairArray->size];
	uint numOfPairs = pairArray->size;
	BinSpherePair *pairs = pairArray->objects;

	uint lastBinID = pairs[0].binID;
	uint dataCnt = 0;
	data[dataCnt].binID = lastBinID;
	data[dataCnt].startIndex = 0;
	for (uint i = 1; i < numOfPairs; i++) {
		if (pairs[i].binID != lastBinID) {
			if (i - data[dataCnt].startIndex > 1) {
				data[dataCnt].numOfObjects = i - data[dataCnt].startIndex;
				dataCnt++;
			}

			lastBinID = pairs[i].binID;
			data[dataCnt].binID = lastBinID;
			data[dataCnt].startIndex = i;
		}
	}
	/* last */
	if (numOfPairs - data[dataCnt].startIndex > 1) {
		data[dataCnt].numOfObjects = numOfPairs - data[dataCnt].startIndex;
		dataCnt++;
	}

	BinSphereData *auxArr = new BinSphereData[dataCnt];
	MC_RadixSort_32_16(data, numOfObjects, auxArr, dataCnt);

	memcpy(auxArr, data, dataCnt * sizeof(BinSphereData));
	delete[] data;

	BinSphereDataArray *dataArray = new BinSphereDataArray;
	dataArray->size = dataCnt;
	dataArray->objects = auxArr;

	return dataArray;
}

void stage7(SphereArray* sphereArray, BinSpherePairArray* pairArray, BinSphereDataArray *dataArray, float binSize) {
	int numDetected = 0;
	Sphere *spheres = sphereArray->objects;
	BinSpherePair *pairs = pairArray->objects;

	float halfBinRate = 0.5 / binSize;

	for (int i = 0; i < dataArray->size; i++) {
		uint binID = dataArray->objects[i].binID;
		uint startIndex = dataArray->objects[i].startIndex;
		uint endIndex = startIndex + dataArray->objects[i].numOfObjects;

		for (uint objA = startIndex; objA < endIndex; objA++) {
			for (uint objB = objA + 1; objB < endIndex; objB++) {
				Sphere &sphereA = spheres[pairs[objA].objectID];
				Sphere &sphereB = spheres[pairs[objB].objectID];

				float distX = sphereA.x - sphereB.x;
				float distY = sphereA.y - sphereB.y;
				float distZ = sphereA.z - sphereB.z;

				float distC = sqrt(distX * distX + distY * distY + distZ * distZ);
				float distR = sphereA.r + sphereB.r;

				if (distC <= distR) {
					float idx = (sphereA.x + sphereB.x) * halfBinRate;
					float idy = (sphereA.y + sphereB.y) * halfBinRate;
					float idz = (sphereA.z + sphereB.z) * halfBinRate;

					#define MC_FloorCast_STG7(v) \
						(v >= 0) ? static_cast<int>(v) : static_cast<int>(v) - 1;

					int x = MC_FloorCast_STG7(idx);
					int y = MC_FloorCast_STG7(idy);
					int z = MC_FloorCast_STG7(idz);

					uint key = MC_Key(x, y, z);

					if (key == binID) {
						numDetected++;
					}
				}
			}
		}
	}

	printf("numOfCollisions = %d\n", numDetected);
}
