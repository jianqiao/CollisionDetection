#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <thrust/scan.h>
#include <thrust/sort.h>

#include "utils.h"
#include "coldetect.h"
#include "types.h"

/**
 * This macro checks return value of the CUDA runtime call and exits
 * the application if the call failed.
 */
#define CUDA_CHECK_RETURN(value) {											\
	cudaError_t _m_cudaStat = value;										\
	if (_m_cudaStat != cudaSuccess) {										\
		fprintf(stderr, "Error %s at line %d in file %s\n",					\
				cudaGetErrorString(_m_cudaStat), __LINE__, __FILE__);		\
		exit(1);															\
	} }

#define CUDA_CHECK_MALLOC(var, type, size) \
	CUDA_CHECK_RETURN(cudaMalloc((void**) &var, sizeof(type) * size));

#define CUDA_CHECK_COPY(varDst, varSrc, type, size, dir) \
	CUDA_CHECK_RETURN(cudaMemcpy(varDst, varSrc, sizeof(type) * size, dir));

#define CUDA_CHECK_FREE(var) \
	CUDA_CHECK_RETURN(cudaFree((void*) var));


__device__ void checkPair(Sphere &sphereA, Sphere &sphereB,
						  int binIDs, float halfBinRate,
						  int &numDetected) {

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

		if (key == binIDs) {
			numDetected++;
		}
	}
}

/*
__global__ void checkCollision0(Sphere *spheres, BinSpherePair *pairs,
							   BinSphereData *data, int *detectCount,
							   int numOfThreads, float halfBinRate) {
	int threadId = blockIdx.x * blockDim.x + threadIdx.x;
	int numDetected = 0;

	if (threadId < numOfThreads) {
		uint binIDs = data[threadId].binIDs;
		uint numOfObjects = data[threadId].numOfObjects;
		uint startIndex = data[threadId].startIndex;

		for (uint objA = 1; objA < numOfObjects; objA++) {
			Sphere sphereA = spheres[pairs[startIndex+objA].objectID];

			for (uint objB = 0; objB < objA; objB++) {
				Sphere sphereB = spheres[pairs[startIndex+objB].objectID];

				checkPair(sphereA, sphereB, binIDs, halfBinRate, numDetected);
			}
		}
		detectCount[threadId] = numDetected;
	}
}


__global__ void checkCollision2(Sphere *spheres, BinSpherePair *pairs,
					   BinSphereData *data, int *detectCount,
					   int numOfThreads, float halfBinRate) {
	int threadId = blockIdx.x * blockDim.x + threadIdx.x;
	int numDetected = 0;

	if (threadId < numOfThreads) {
		Sphere s00, s01, s02, s03,
			   s04, s05, s06, s07,
			   s08, s09, s10, s11;

		uint binIDs = data[threadId].binIDs;
		uint numOfObjects = data[threadId].numOfObjects;
		uint startIndex = data[threadId].startIndex;

		#define MC_SetValue(s, i) \
			s = spheres[pairs[startIndex+i].objectID];

		#define MC_CheckPair(sphereA, sphereB) \
			checkPair(sphereA, sphereB, \
					  binIDs, halfBinRate, numDetected);

		#define MC_CheckPoint(i) \
			if (numOfObjects < i) goto RETURN;

		MC_SetValue(s00, 0);
		MC_SetValue(s01, 1);
		MC_CheckPair(s00, s01);

		MC_CheckPoint(3);
		MC_SetValue(s02, 2);
		MC_CheckPair(s00, s02);
		MC_CheckPair(s01, s02);

		MC_CheckPoint(4);
		MC_SetValue(s03, 3);
		MC_CheckPair(s00, s03);
		MC_CheckPair(s01, s03);
		MC_CheckPair(s02, s03);

		MC_CheckPoint(5);
		MC_SetValue(s04, 4);
		MC_CheckPair(s00, s04);
		MC_CheckPair(s01, s04);
		MC_CheckPair(s02, s04);
		MC_CheckPair(s03, s04);

		MC_CheckPoint(6);
		MC_SetValue(s05, 5);
		MC_CheckPair(s00, s05);
		MC_CheckPair(s01, s05);
		MC_CheckPair(s02, s05);
		MC_CheckPair(s03, s05);
		MC_CheckPair(s04, s05);

		MC_CheckPoint(7);
		MC_SetValue(s06, 6);
		MC_CheckPair(s00, s06);
		MC_CheckPair(s01, s06);
		MC_CheckPair(s02, s06);
		MC_CheckPair(s03, s06);
		MC_CheckPair(s04, s06);
		MC_CheckPair(s05, s06);

		MC_CheckPoint(8);
		MC_SetValue(s07, 7);
		MC_CheckPair(s00, s07);
		MC_CheckPair(s01, s07);
		MC_CheckPair(s02, s07);
		MC_CheckPair(s03, s07);
		MC_CheckPair(s04, s07);
		MC_CheckPair(s05, s07);
		MC_CheckPair(s06, s07);

		MC_CheckPoint(9);
		MC_SetValue(s08, 8);
		MC_CheckPair(s00, s08);
		MC_CheckPair(s01, s08);
		MC_CheckPair(s02, s08);
		MC_CheckPair(s03, s08);
		MC_CheckPair(s04, s08);
		MC_CheckPair(s05, s08);
		MC_CheckPair(s06, s08);
		MC_CheckPair(s07, s08);

		MC_CheckPoint(10);
		MC_SetValue(s09, 9);
		MC_CheckPair(s00, s09);
		MC_CheckPair(s01, s09);
		MC_CheckPair(s02, s09);
		MC_CheckPair(s03, s09);
		MC_CheckPair(s04, s09);
		MC_CheckPair(s05, s09);
		MC_CheckPair(s06, s09);
		MC_CheckPair(s07, s09);
		MC_CheckPair(s08, s09);

		MC_CheckPoint(11);
		MC_SetValue(s10, 10);
		MC_CheckPair(s00, s10);
		MC_CheckPair(s01, s10);
		MC_CheckPair(s02, s10);
		MC_CheckPair(s03, s10);
		MC_CheckPair(s04, s10);
		MC_CheckPair(s05, s10);
		MC_CheckPair(s06, s10);
		MC_CheckPair(s07, s10);
		MC_CheckPair(s08, s10);
		MC_CheckPair(s09, s10);

		MC_CheckPoint(12);
		MC_SetValue(s11, 11);
		MC_CheckPair(s00, s11);
		MC_CheckPair(s01, s11);
		MC_CheckPair(s02, s11);
		MC_CheckPair(s03, s11);
		MC_CheckPair(s04, s11);
		MC_CheckPair(s05, s11);
		MC_CheckPair(s06, s11);
		MC_CheckPair(s07, s11);
		MC_CheckPair(s08, s11);
		MC_CheckPair(s09, s11);
		MC_CheckPair(s10, s11);

	RETURN:
		detectCount[threadId] = numDetected;
	}
}
*/

__global__ void checkCollision(Sphere *spheres, BinSpherePair *pairs,
							   BinSphereData *data, int *detectCount,
							   int numOfThreads, float halfBinRate) {
	int threadId = blockIdx.x * blockDim.x + threadIdx.x;
	int numDetected = 0;

	if (threadId < numOfThreads) {
		uint binIDs = data[threadId].binID;
		uint numOfObjects = data[threadId].numOfObjects;
		uint startIndex = data[threadId].startIndex;

		Sphere localSpheres[64];
		for (uint i = 0; i < numOfObjects; i++) {
			localSpheres[i] = spheres[pairs[startIndex+i].objectID];
		}

		for (uint objA = 1; objA < numOfObjects; objA++) {
			Sphere sphereA = localSpheres[objA];

			for (uint objB = 0; objB < objA; objB++) {
				Sphere sphereB = localSpheres[objB];

				checkPair(sphereA, sphereB, binIDs, halfBinRate, numDetected);
			}
		}
		detectCount[threadId] = numDetected;
	}
}

extern TimeRecorder<void> timer;

void gpu_stage7(SphereArray* sphereArray, BinSpherePairArray* pairArray, BinSphereDataArray *dataArray, float binSize) {
	Sphere			*_spheres;
	BinSpherePair	*_pairs;
	BinSphereData	*_data;
	int				*_detectCount;

	cudaDeviceSetCacheConfig( cudaFuncCachePreferL1 );

	CUDA_CHECK_RETURN(cudaMalloc((void**) &_spheres, sizeof(Sphere) * sphereArray->size));
	CUDA_CHECK_RETURN(cudaMalloc((void**) &_pairs, sizeof(BinSpherePair) * pairArray->size));
	CUDA_CHECK_RETURN(cudaMalloc((void**) &_data, sizeof(BinSphereData) * dataArray->size));
	CUDA_CHECK_RETURN(cudaMalloc((void**) &_detectCount, sizeof(int) * dataArray->size));

	CUDA_CHECK_RETURN(cudaMemcpy(_spheres, sphereArray->objects, sizeof(Sphere) * sphereArray->size, cudaMemcpyHostToDevice));
	CUDA_CHECK_RETURN(cudaMemcpy(_pairs, pairArray->objects, sizeof(BinSpherePair) * pairArray->size, cudaMemcpyHostToDevice));
	CUDA_CHECK_RETURN(cudaMemcpy(_data, dataArray->objects, sizeof(BinSphereData) * dataArray->size, cudaMemcpyHostToDevice));
/*
	float *_coalescedStore;
	int lineSize = ((dataArray->size + 31) / 32) * 32;
	CUDA_CHECK_RETURN(cudaMalloc((void**) &_coalescedStore, sizeof(Sphere) * lineSize * 32));
*/
	timer.appendStart("kernel");

	float halfBinRate = 0.5 / binSize;
	int numOfThreads = dataArray->size;

	int threadsPerBlock = 64;
	int numOfBlocks = (numOfThreads + threadsPerBlock - 1) / threadsPerBlock;

	checkCollision<<<numOfBlocks, threadsPerBlock>>>(
				_spheres, _pairs,
				_data, _detectCount,
				dataArray->size, halfBinRate
	);

	CUDA_CHECK_RETURN(cudaThreadSynchronize());	// Wait for the GPU launched work to complete
	CUDA_CHECK_RETURN(cudaGetLastError());

	timer.appendEnd("kernel");
	printf("kernel time: %lf\n", timer.getTotalTime("kernel"));

	int *detectCount = new int[numOfThreads];
	CUDA_CHECK_RETURN(cudaMemcpy(detectCount, _detectCount, sizeof(int) * numOfThreads, cudaMemcpyDeviceToHost));

	int numOfCollisions = 0;
	for (int i = 0; i < numOfThreads; i++)
		numOfCollisions += detectCount[i];

	printf("number of collisions = %d\n", numOfCollisions);

	CUDA_CHECK_RETURN(cudaFree((void*) _spheres));
	CUDA_CHECK_RETURN(cudaFree((void*) _pairs));
	CUDA_CHECK_RETURN(cudaFree((void*) _data));
	CUDA_CHECK_RETURN(cudaFree((void*) _detectCount));

	CUDA_CHECK_RETURN(cudaDeviceReset());
}


__global__ void BinningStep1(Sphere *spheres, BinSphereBound *bounds, int *numOfBins,
		                	 float binRate, int numOfSpheres, int spheresPerThread) {
	int threadId = blockIdx.x * blockDim.x + threadIdx.x;
	int index = ((threadId / 32) * 32) * spheresPerThread + (threadId % 32);
	int endIndex = min(numOfSpheres, index + spheresPerThread * 32);

	while (index < endIndex) {
		Sphere sphere = spheres[index];
		BinSphereBound bound;

		float idx = sphere.x * binRate;
		float idy = sphere.y * binRate;
		float idz = sphere.z * binRate;
		float idr = sphere.r * binRate;

		bound.minIDX = __float2int_rd(idx - idr);
		bound.minIDY = __float2int_rd(idy - idr);
		bound.minIDZ = __float2int_rd(idz - idr);

		bound.maxIDX = __float2int_rd(idx + idr);
		bound.maxIDY = __float2int_rd(idy + idr);
		bound.maxIDZ = __float2int_rd(idz + idr);

		int size = (bound.maxIDX - bound.minIDX + 1) *
				   (bound.maxIDY - bound.minIDY + 1) *
				   (bound.maxIDZ - bound.minIDZ + 1);

		bounds[index]		= bound;
		numOfBins[index]	= size;

		index += 32;
	}
}

__global__ void BinningStep2(BinSphereBound *bounds, int *numOfBins,
							 uint *binIDs, uint *sphereIDs,
							 int numOfSpheres, int spheresPerThread) {

	int threadId = blockIdx.x * blockDim.x + threadIdx.x;

	int index = ((threadId / 32) * 32) * spheresPerThread + (threadId % 32);
	int endIndex = min(numOfSpheres, index + spheresPerThread * 32);

	while (index < endIndex) {
		int cnt = numOfBins[index];

		BinSphereBound bound = bounds[index];
		for (int idx = bound.minIDX; idx <= bound.maxIDX; idx++) {
			for (int idy = bound.minIDY; idy <= bound.maxIDY; idy++) {
				for (int idz = bound.minIDZ; idz <= bound.maxIDZ; idz++) {
					uint key = MC_Key(idx, idy, idz);
					binIDs[cnt]		= key;
					sphereIDs[cnt]	= index;
					cnt++;
				}
			}
		}

		index += 32;
	}
}

extern TimeRecorder<void> timer;

void binningStep1(Sphere *_spheres, int numOfSpheres, float binSize,
				  int *_numOfBins, BinSphereBound *_bounds) {

	int threadsPerBlock		= 128;

	int maxNumOfBlocks		= 4096;
	int maxNumOfThreads		= maxNumOfBlocks * threadsPerBlock;
	int spheresPerThread	= MC_CeilDivide(numOfSpheres, maxNumOfThreads);
	int spheresPerBlock		= threadsPerBlock * spheresPerThread;
	int numOfBlocks = MC_CeilDivide(numOfSpheres, spheresPerBlock);

	timer.appendStart("binningStep1");

	float binRate = 1.0 / binSize;

	BinningStep1<<<numOfBlocks, threadsPerBlock>>>(
				_spheres, _bounds, _numOfBins,
				binRate, numOfSpheres, spheresPerThread);

	CUDA_CHECK_RETURN(cudaThreadSynchronize());	// Wait for the GPU launched work to complete
	CUDA_CHECK_RETURN(cudaGetLastError());

	timer.appendEnd("binningStep1");
	printf("binning step 1 time = %lf\n", timer.getTotalTime("binningStep1"));
}

void binningStep2(BinSphereBound *_bounds, int *_numOfBins,
				  uint *_binIDs, uint *_sphereIDs, int numOfSpheres) {

	int threadsPerBlock		= 128;

	int maxNumOfBlocks		= 4096;
	int maxNumOfThreads		= maxNumOfBlocks * threadsPerBlock;
	int spheresPerThread	= MC_CeilDivide(numOfSpheres, maxNumOfThreads);
	int spheresPerBlock		= threadsPerBlock * spheresPerThread;
	int numOfBlocks = MC_CeilDivide(numOfSpheres, spheresPerBlock);

	timer.appendStart("binningStep2");

	BinningStep2<<<numOfBlocks, threadsPerBlock>>>(
				_bounds, _numOfBins,
				_binIDs, _sphereIDs,
				numOfSpheres, spheresPerThread);

	CUDA_CHECK_RETURN(cudaThreadSynchronize());	// Wait for the GPU launched work to complete
	CUDA_CHECK_RETURN(cudaGetLastError());

	timer.appendEnd("binningStep2");
	printf("binning step 2 time = %lf\n", timer.getTotalTime("binningStep2"));
}

BinSpherePairArray* gpu_stage1To4(SphereArray *sphereArray, float binSize) {
	Sphere			*_spheres;
	int				*_numOfBins;
	BinSphereBound	*_bounds;

	cudaDeviceSetCacheConfig( cudaFuncCachePreferL1 );

	int numOfSpheres = sphereArray->size;
	CUDA_CHECK_MALLOC(_spheres,		Sphere,				numOfSpheres);
	CUDA_CHECK_MALLOC(_bounds,		BinSphereBound,		numOfSpheres);
	CUDA_CHECK_MALLOC(_numOfBins,	int,				numOfSpheres);
	CUDA_CHECK_COPY(_spheres, sphereArray->objects, Sphere, numOfSpheres, cudaMemcpyHostToDevice);

	timer.appendStart("binningKernel");

	binningStep1(_spheres, numOfSpheres, binSize, _numOfBins, _bounds);

	thrust::device_ptr<int> _numOfBinsArray(_numOfBins);
	int lastBinSize	= _numOfBinsArray[numOfSpheres - 1];

	thrust::exclusive_scan (_numOfBinsArray , _numOfBinsArray + numOfSpheres , _numOfBinsArray);
	int numOfPairs	= _numOfBinsArray[numOfSpheres - 1] + lastBinSize;

	uint *_binIDs;
	uint *_sphereIDs;

	CUDA_CHECK_MALLOC(_binIDs,		uint,		numOfPairs);
	CUDA_CHECK_MALLOC(_sphereIDs,	uint,		numOfPairs);
//	printf("numOfPairs = %d\n", numOfPairs);

	binningStep2(_bounds, _numOfBinsArray.get(), _binIDs, _sphereIDs, numOfSpheres) ;

	thrust::device_ptr<uint> _binIDArray(_binIDs);
	thrust::device_ptr<uint> _sphereIDArray(_sphereIDs);
	thrust::stable_sort_by_key(_binIDArray, _binIDArray + numOfPairs, _sphereIDArray);

	timer.appendEnd("binningKernel");
	printf("binning total kernel time: %lf\n", timer.getTotalTime("binningKernel"));

	uint *binIDs			= new uint[numOfPairs];
	uint *sphereIDs			= new uint[numOfPairs];
	BinSpherePair *pairs	= new BinSpherePair[numOfPairs];

	CUDA_CHECK_COPY(binIDs, _binIDArray.get(), uint, numOfPairs, cudaMemcpyDeviceToHost);
	CUDA_CHECK_COPY(sphereIDs, _sphereIDArray.get(), uint, numOfPairs, cudaMemcpyDeviceToHost);

	for (int i = 0; i < numOfPairs; i++) {
		pairs[i].binID = binIDs[i];
		pairs[i].objectID = sphereIDs[i];
	}
	printf("numOfPairs = %d\n", numOfPairs);

	delete []binIDs;
	delete []sphereIDs;

	CUDA_CHECK_FREE(_spheres);
	CUDA_CHECK_FREE(_bounds);
	CUDA_CHECK_FREE(_numOfBins);
	CUDA_CHECK_FREE(_binIDs);
	CUDA_CHECK_FREE(_sphereIDs);

	BinSpherePairArray *pairArray = new BinSpherePairArray;
	pairArray->size = numOfPairs;
	pairArray->objects = pairs;

	return pairArray;
}

