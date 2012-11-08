/* *
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

#include "types.h"
#include "utils.h"

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

__global__ void checkCollision(Sphere *spheres, BinSpherePair *pairs,
							   BinSphereData *data, int *detectCount,
							   int numOfThreads, float halfBinRate) {
	int threadId = blockIdx.x * blockDim.x + threadIdx.x;
	int numDetected = 0;

	if (threadId < numOfThreads) {
		uint binID = data[threadId].binID;
		uint numOfObjects = data[threadId].numOfObjects;
		uint startIndex = data[threadId].startIndex;

		Sphere localSpheres[32];
		for (uint i = 0; i < numOfObjects; i++) {
			localSpheres[i] = spheres[pairs[startIndex+i].objectID];
		}

		for (uint objA = 0; objA < numOfObjects; objA++) {
			Sphere sphereA = localSpheres[objA];
//			Sphere sphereA = spheres[threadId];

			for (uint objB = objA + 1; objB < numOfObjects; objB++) {
				Sphere sphereB = localSpheres[objB];
//				Sphere sphereB = spheres[threadId+4096];

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
		detectCount[threadId] = numDetected;
	}
}

extern TimeRecorder<void> timer;

void gpu_stage7(SphereArray* sphereArray, BinSpherePairArray* pairArray, BinSphereDataArray *dataArray, float binSize) {
	Sphere			*_spheres;
	BinSpherePair	*_pairs;
	BinSphereData	*_data;
	int				*_detectCount;

	CUDA_CHECK_RETURN(cudaMalloc((void**) &_spheres, sizeof(Sphere) * sphereArray->size));
	CUDA_CHECK_RETURN(cudaMalloc((void**) &_pairs, sizeof(BinSpherePair) * pairArray->size));
	CUDA_CHECK_RETURN(cudaMalloc((void**) &_data, sizeof(BinSphereData) * dataArray->size));
	CUDA_CHECK_RETURN(cudaMalloc((void**) &_detectCount, sizeof(int) * dataArray->size));

	CUDA_CHECK_RETURN(cudaMemcpy(_spheres, sphereArray->objects, sizeof(Sphere) * sphereArray->size, cudaMemcpyHostToDevice));
	CUDA_CHECK_RETURN(cudaMemcpy(_pairs, pairArray->objects, sizeof(BinSpherePair) * pairArray->size, cudaMemcpyHostToDevice));
	CUDA_CHECK_RETURN(cudaMemcpy(_data, dataArray->objects, sizeof(BinSphereData) * dataArray->size, cudaMemcpyHostToDevice));

	float halfBinRate = 0.5 / binSize;
	int numOfThreads = dataArray->size;

	int threadsPerBlock = 128;
	int numOfBlocks = (numOfThreads + threadsPerBlock - 1) / threadsPerBlock;

	timer.appendStart("kernel");

	checkCollision<<<numOfBlocks, threadsPerBlock>>>(
				_spheres, _pairs,
				_data, _detectCount,
				dataArray->size, halfBinRate
	);

	CUDA_CHECK_RETURN(cudaThreadSynchronize());	// Wait for the GPU launched work to complete
	CUDA_CHECK_RETURN(cudaGetLastError());
	timer.appendEnd("kernel");

	int *detectCount = new int[numOfThreads];
	CUDA_CHECK_RETURN(cudaMemcpy(detectCount, _detectCount, sizeof(int) * numOfThreads, cudaMemcpyDeviceToHost));

	printf("kernel time: %lf\n", timer.getTotalTime("kernel"));

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


/*
int main(void) {
	void *d = NULL;
	int i;
	unsigned int idata[WORK_SIZE], odata[WORK_SIZE];

	for (i = 0; i < WORK_SIZE; i++)
		idata[i] = (unsigned int) i;

	CUDA_CHECK_RETURN(cudaMalloc((void**) &d, sizeof(int) * WORK_SIZE));
	CUDA_CHECK_RETURN(
			cudaMemcpy(d, idata, sizeof(int) * WORK_SIZE, cudaMemcpyHostToDevice));

	bitreverse<<<1, WORK_SIZE, WORK_SIZE * sizeof(int)>>>(d);

	CUDA_CHECK_RETURN(cudaThreadSynchronize());	// Wait for the GPU launched work to complete
	CUDA_CHECK_RETURN(cudaGetLastError());
	CUDA_CHECK_RETURN(cudaMemcpy(odata, d, sizeof(int) * WORK_SIZE, cudaMemcpyDeviceToHost));

	for (i = 0; i < WORK_SIZE; i++)
		printf("Input value: %u, device output: %u, host output: %u\n",
				idata[i], odata[i], bitreverse(idata[i]));

	CUDA_CHECK_RETURN(cudaFree((void*) d));
	CUDA_CHECK_RETURN(cudaDeviceReset());

	return 0;
}
*/
