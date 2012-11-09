
/**
 * Copyright 1993-2012 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 */
#include <stdio.h>
#include <stdlib.h>

#include <cuda.h>

#include "types.h"
#include "utils.h"
#include <thrust/scan.h>

/**
 * This macro checks return value of the CUDA runtime call and exits
 * the application if the call failed.
 *
 * See cuda.h for error code descriptions.
 */
#define CHECK_CUDA_RESULT(N) {											\
	CUresult result = N;												\
	if (result != 0) {													\
		printf("CUDA call on line %d returned error %d\n", __LINE__,	\
			result);													\
		exit(1);														\
	} }

#define CUDA_CHECK_RETURN(value) {											\
	cudaError_t _m_cudaStat = value;										\
	if (_m_cudaStat != cudaSuccess) {										\
		fprintf(stderr, "Error %s at line %d in file %s\n",					\
				cudaGetErrorString(_m_cudaStat), __LINE__, __FILE__);		\
		exit(1);															\
	} }

__global__ void Binning(Sphere *spheres, int binSize, BinSphereBound *bounds, int numOfSpheres,
						int *numOfBins,int numOfThreads	)
{

	int threadId = blockIdx.x * blockDim.x + threadIdx.x;
	float binRate = 1.0 / binSize;
	if(threadId < numOfThreads){
	for (int i = (threadId)*10; i < (threadId+1)*10; i++) {
		Sphere &sphere = spheres[i];
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
		numOfBins[i]= (bounds[i].maxIDX - bounds[i].minIDX + 1) *
									  (bounds[i].maxIDY - bounds[i].minIDY + 1) *
									  (bounds[i].maxIDZ - bounds[i].minIDZ + 1);

	}
	}



}

__global__ void Binning1(int binSize, BinSphereBound *bounds, int numOfSpheres,
						int *numOfBins,BinSpherePair *_pairs, int numOfThreads	)
{
	int threadId = blockIdx.x * blockDim.x + threadIdx.x;
	//int numOfPairs=numOfBins[numOfSpheres-1];
	//printf("%d\n",numOfPairs);
		if(threadId < numOfThreads){


			for (int i = (threadId)*10; i < (threadId+1)*10; i++) {
				int cnt = 0;
				BinSphereBound &bound = bounds[i];
				for (int idx = bound.minIDX; idx <= bound.maxIDX; idx++) {
					for (int idy = bound.minIDY; idy <= bound.maxIDY; idy++) {
						for (int idz = bound.minIDZ; idz <= bound.maxIDZ; idz++) {
							uint key = MC_Key(idx, idy, idz);
							if(i!=0){
							//if(threadId==799)printf("%d\n",numOfBins[i-1]);
							_pairs[numOfBins[i-1]+cnt].objectID	= i;
							_pairs[numOfBins[i-1]+cnt].binID	= key;
							}
							else{
								_pairs[cnt].objectID	= i;
								_pairs[cnt].binID	= key;
							}
							cnt++;
						//printf("[%u] %d, %d, %d, key = %u\n", i, idx, idy, idz, key);
						}
					}
				}
			}
		}
}


extern TimeRecorder<void> timer;

BinSpherePairArray* gpu_stage1To3(SphereArray* sphereArray,
						      float binSize)
{
	Sphere	*_spheres;
	BinSphereBound *_bounds;
	int *_numOfBins;
	BinSpherePair *_pairs;

    //printf("1.1\n");

	BinSphereBound *bounds = new BinSphereBound[sphereArray->size];
	int *numOfBins = new int[sphereArray->size];
	CUDA_CHECK_RETURN(cudaMalloc((void**) &_spheres, sizeof(Sphere) * sphereArray->size));
	CUDA_CHECK_RETURN(cudaMemcpy(_spheres, sphereArray->objects, sizeof(Sphere) * sphereArray->size, cudaMemcpyHostToDevice));
	CUDA_CHECK_RETURN(cudaMalloc((void**) &_bounds, sizeof(BinSphereBound) * sphereArray->size));
	CUDA_CHECK_RETURN(cudaMalloc((void**) &_numOfBins, sizeof(int) * sphereArray->size));

	int numOfSpheres=sphereArray->size;
	int numOfThreads=(sphereArray->size/10);
	int threadsPerBlock = 128;
	int numOfBlocks = (numOfThreads + threadsPerBlock - 1) / threadsPerBlock;

	timer.appendStart("kernel_stage1");

	Binning<<<numOfBlocks, threadsPerBlock>>>(_spheres,
										binSize, _bounds
										,numOfSpheres,_numOfBins,numOfThreads);

	CUDA_CHECK_RETURN(cudaMemcpy(bounds, _bounds, sizeof(BinSphereBound) * sphereArray->size,cudaMemcpyDeviceToHost));
	CUDA_CHECK_RETURN(cudaMemcpy(numOfBins, _numOfBins, sizeof(int) * sphereArray->size,cudaMemcpyDeviceToHost));

	thrust :: inclusive_scan (numOfBins,numOfBins + sphereArray->size ,numOfBins);

	timer.appendEnd("kernel_stage1");
	printf("%lf\n", timer.getTotalTime("kernel_stage1"));
    int numOfPairs=numOfBins[sphereArray->size-1];
    //printf("%d\n",numOfPairs);


	BinSpherePair *pairs = new BinSpherePair[numOfPairs];
	CUDA_CHECK_RETURN(cudaMalloc((void**) &_pairs, sizeof(BinSpherePair) * numOfPairs));
	CUDA_CHECK_RETURN(cudaMemcpy(_numOfBins, numOfBins, sizeof(int) * sphereArray->size,cudaMemcpyHostToDevice));
	Binning1<<<numOfBlocks, threadsPerBlock>>>(binSize, _bounds,
											   numOfSpheres,_numOfBins,_pairs, numOfThreads); //numOfThreads);
	CUDA_CHECK_RETURN(cudaMemcpy(pairs, _pairs, sizeof(BinSpherePair) * numOfPairs,cudaMemcpyDeviceToHost));

    /*
	int cnt = 0;
	BinSpherePair *pairs = new BinSpherePair[numOfPairs];
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
	*/
	BinSpherePairArray *pairArray = new BinSpherePairArray;
	pairArray->size =numOfPairs;
	pairArray->objects = pairs;
	return pairArray;
	/*
	BinSpherePair *pairs = new BinSpherePair[numOfPairs];

	Binning1<<<numOfBlocks, threadsPerBlock>>>(binSize, _bounds
											,numOfSpheres,_numOfBins);

		BinSpherePairArray *pairArray = new BinSpherePairArray;
		pairArray->size = numOfPairs;
		pairArray->objects = pairs;

		return pairArray;
		*/
}
