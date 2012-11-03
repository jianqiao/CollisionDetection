/*
 * init.C
 *
 *  Created on: Nov 2, 2012
 *      Author: jianqiao
 */

#include "types.h"
#include "stages.h"
#include "utils.h"
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



BinSpherePairArray* stage1To3(SphereArray* sphereArray,
						      float binSize) {
	struct BinSphereBound {
		int minIDX, minIDY, minIDZ;
		int maxIDX, maxIDY, maxIDZ;
	};

	float binRate = 1.0 / binSize;
	int minIDX = 0x7FFFFFFF, minIDY = 0x7FFFFFFF, minIDZ = 0x7FFFFFFF;
	int maxIDX = 0x80000000, maxIDY = 0x80000000, maxIDZ = 0x80000000;

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

		#define MC_UpdateMinMax(minItem, maxItem) \
			if (bound.minItem < minItem)		minItem = bound.minItem; \
			else if (bound.maxItem > maxItem)	maxItem = bound.maxItem;

		MC_UpdateMinMax(minIDX, maxIDX);
		MC_UpdateMinMax(minIDY, maxIDY);
		MC_UpdateMinMax(minIDZ, maxIDZ);
	}

	int bitwidthIDX = 32 - __builtin_clz(maxIDX - minIDX);
	int bitwidthIDY = 32 - __builtin_clz(maxIDY - minIDY);
	int bitwidthIDZ = 32 - __builtin_clz(maxIDZ - minIDZ);

	int shiftX = bitwidthIDY + bitwidthIDZ;
	int shiftY = bitwidthIDZ;
	// int shiftZ = 0;

	BinSpherePair *pairs = new BinSpherePair[numOfPairs];
	int cnt = 0;
	for (uint i = 0; i < numOfSpheres; i++) {
		BinSphereBound &bound = bounds[i];
		for (int idx = bound.minIDX - minIDX,
				IDX_END = bound.maxIDX - minIDX; idx <= IDX_END; idx++) {
			for (int idy = bound.minIDY - minIDY,
					IDY_END = bound.maxIDY - minIDY; idy <= IDY_END; idy++) {
				for (int idz = bound.minIDZ - minIDZ,
						IDZ_END = bound.maxIDZ - minIDZ; idz <= IDZ_END; idz++) {

					uint key = (idx << shiftX) | (idy << shiftY) | idz;
					pairs[cnt].objectID	= i;
					pairs[cnt].binID	= key;
					cnt++;
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
}

BinSphereDataArray *stage5(BinSpherePairArray *pairArray) {
	BinSphereData *data = new BinSphereData[pairArray->size];
	uint numOfPairs = pairArray->size;
	BinSpherePair *pairs = pairArray->objects;

	uint lastBinID = pairs[0].binID;
	uint dataCnt = 0;
	data[0].startIndex = 0;
	for (uint i = 1; i < numOfPairs; i++) {
		if (pairs[i].binID != lastBinID) {
			if (i - data[dataCnt].startIndex > 1) {
				data[dataCnt].endIndex = i;
				dataCnt++;
			}

			lastBinID = pairs[i].binID;
			data[dataCnt].binID = lastBinID;
			data[dataCnt].startIndex = i;
		}
	}
	/* last */
	if (numOfPairs - data[dataCnt].startIndex > 1) {
		data[dataCnt].endIndex = numOfPairs;
		dataCnt++;
	}

	BinSphereData *auxArr = new BinSphereData[dataCnt];
	MC_RadixSort_32_16(data, binID, auxArr, dataCnt);

	memcpy(auxArr, data, dataCnt * sizeof(BinSphereData));
	delete[] data;

	BinSphereDataArray *dataArray = new BinSphereDataArray;
	dataArray->size = dataCnt;
	dataArray->objects = auxArr;

	return dataArray;
}
