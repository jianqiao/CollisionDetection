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
	CUDA_CHECK_RETURN(cudaMalloc((void**) &(var), sizeof(type) * size));

#define CUDA_CHECK_COPY(varDst, varSrc, type, size, dir) \
	CUDA_CHECK_RETURN(cudaMemcpyAsync(varDst, varSrc, sizeof(type) * size, dir, stream)); \
	cudaStreamSynchronize(stream);

#define CUDA_CHECK_FREE(var) \
	CUDA_CHECK_RETURN(cudaFree((void*) var));

#define MAX_OBJECTS_PER_BIN 256

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

__global__ void BinDataStep1(uint *binIDs, uint *startIndices, uint *endIndices,
							 int numOfPairs, int pairsPerThread) {

	int threadId = blockIdx.x * blockDim.x + threadIdx.x;

	int index = ((threadId / 32) * 32) * pairsPerThread + (threadId % 32);
	int endIndex = min(numOfPairs, index + pairsPerThread * 32);

	while (index < endIndex) {
		uint currID = binIDs[index];
		uint predID = (index == 0) ? (currID - 1) : binIDs[index-1];
		uint succID = (index == endIndex - 1) ? (currID + 1) : binIDs[index+1];

		startIndices[index]	= ((currID != predID) && (currID == succID));
		endIndices[index]	= ((currID == predID) && (currID != succID));

		index += 32;
	}
}

__global__ void BinDataStep2(uint *binIDs, uint *startIndices, GPUBinData *binData,
							 int numOfPairs, int pairsPerThread) {

	int threadId = blockIdx.x * blockDim.x + threadIdx.x;

	int index = ((threadId / 32) * 32) * pairsPerThread + (threadId % 32);
	int endIndex = min(numOfPairs, index + pairsPerThread * 32);

	while (index < endIndex) {
		uint currID = binIDs[index];
		uint predID = (index == 0) ? (currID - 1) : binIDs[index-1];
		uint succID = (index == endIndex - 1) ? (currID + 1) : binIDs[index+1];

		if ((currID != predID) && (currID == succID)) {
			int targetIndex = startIndices[index];
			binData[targetIndex].binID = currID;
			binData[targetIndex].startIndex = index;
		}

		index += 32;
	}
}

__global__ void BinDataStep3(uint *binIDs, uint *endIndices,
							 GPUBinData *binData, uint *objectCount,
							 int numOfPairs, int pairsPerThread) {

	int threadId = blockIdx.x * blockDim.x + threadIdx.x;

	int index = ((threadId / 32) * 32) * pairsPerThread + (threadId % 32);
	int endIndex = min(numOfPairs, index + pairsPerThread * 32);

	while (index < endIndex) {
		uint currID = binIDs[index];
		uint predID = (index == 0) ? (currID - 1) : binIDs[index-1];
		uint succID = (index == endIndex - 1) ? (currID + 1) : binIDs[index+1];

		if ((currID == predID) && (currID != succID)) {
			int targetIndex = endIndices[index];
			objectCount[targetIndex] = index - binData[targetIndex].startIndex + 1;
		}

		index += 32;
	}
}

__forceinline__ __device__ void checkPair(Sphere sphereA, Sphere sphereB,
										  uint aID, uint bID,
						  	  	  	  	  uint binID, float halfBinRate,
						  	  	  	  	  uint *&pairPtr, int &numDetected) {

	float distX = sphereA.x - sphereB.x;
	float distY = sphereA.y - sphereB.y;
	float distZ = sphereA.z - sphereB.z;

	float distC = sqrt(distX * distX + distY * distY + distZ * distZ);
	float distR = sphereA.r + sphereB.r;

	if (distC <= distR) {
		float idx = (sphereA.x + sphereB.x) * halfBinRate;
		float idy = (sphereA.y + sphereB.y) * halfBinRate;
		float idz = (sphereA.z + sphereB.z) * halfBinRate;

		int x = __float2int_rd(idx);
		int y = __float2int_rd(idy);
		int z = __float2int_rd(idz);

		uint key = MC_Key(x, y, z);

		if (key == binID) {
			++numDetected;
			*pairPtr = (aID << 16) | bID;
			pairPtr += 32;
		}
	}
}

__global__ void checkCollision(Sphere *spheres, uint *sphereIDs,
		                       GPUBinData *binData, uint *objectCount,
		                       int *detectCount, uint *collisionPairs,
		                       int numOfThreads, float halfBinRate) {
	int threadId = blockIdx.x * blockDim.x + threadIdx.x;
	int numDetected = 0;

	uint *pairPtr = collisionPairs + (((threadId / 32) * 32) * MAX_OBJECTS_PER_BIN + (threadId % 32));

	if (threadId < numOfThreads) {
		uint binID = binData[threadId].binID;
		uint startIndex = binData[threadId].startIndex;
		uint numOfObjects = objectCount[threadId];

		Sphere localSpheres[MAX_OBJECTS_PER_BIN];
		for (uint i = 0; i < numOfObjects; i++) {
			localSpheres[i] = spheres[sphereIDs[startIndex+i]];
		}

		uint quot	= numOfObjects / 4;
		uint rem	= numOfObjects % 4;

		#define MC_CheckPair(x, y, xID, yID) \
			checkPair(x, y, xID, yID, binID, halfBinRate, pairPtr, numDetected);

		for(uint i = 0; i < quot; i++) {
			uint aID = 4*i;
			uint bID = 4*i+1;
			uint cID = 4*i+2;
			uint dID = 4*i+3;

			Sphere sphereA = localSpheres[aID];
			Sphere sphereB = localSpheres[bID];
			Sphere sphereC = localSpheres[cID];
			Sphere sphereD = localSpheres[dID];

			MC_CheckPair(sphereA, sphereB, aID, bID);
			MC_CheckPair(sphereA, sphereC, aID, cID);
			MC_CheckPair(sphereB, sphereC, bID, cID);
			MC_CheckPair(sphereD, sphereA, dID, aID);
			MC_CheckPair(sphereD, sphereB, dID, bID);
			MC_CheckPair(sphereD, sphereC, dID, cID);

			for(uint eID = 4*i+4; eID < numOfObjects; eID++) {
				Sphere sphereE = localSpheres[eID];

				MC_CheckPair(sphereE, sphereA, eID, aID);
				MC_CheckPair(sphereE, sphereB, eID, bID);
				MC_CheckPair(sphereE, sphereC, eID, cID);
				MC_CheckPair(sphereE, sphereD, eID, dID);
			}
		}

		if(rem > 1) {
			uint aID = numOfObjects - 1;
			uint bID = numOfObjects - 2;
			Sphere sphereA = localSpheres[aID];
			Sphere sphereB = localSpheres[bID];

			MC_CheckPair(sphereA, sphereB, aID, bID);

			if (rem > 2) {
				uint cID = numOfObjects - 3;
				Sphere sphereC = localSpheres[cID];

				MC_CheckPair(sphereC, sphereA, cID, aID);
				MC_CheckPair(sphereC, sphereB, cID, bID);
			}
		}

		detectCount[threadId] = numDetected;
	}
}


__device__ void computePair(Sphere sphereA, Sphere sphereB,
		                    uint sphereAID, uint sphereBID,
						    int pos, CollisionInfo *collisionInfo) {

	CollisionInfo &info = collisionInfo[pos];

	float distX = sphereA.x - sphereB.x;
	float distY = sphereA.y - sphereB.y;
	float distZ = sphereA.z - sphereB.z;

	info.a_id = sphereAID;
	info.b_id = sphereBID;

	float rcpDistC = rsqrt(distX * distX + distY * distY + distZ * distZ);
	float negHalfRcpDistC = -0.5 * rcpDistC;

	info.normal_x = (sphereA.x + sphereB.x) * negHalfRcpDistC;
	info.normal_y = (sphereA.y + sphereB.y) * negHalfRcpDistC;
	info.normal_z = (sphereA.z + sphereB.z) * negHalfRcpDistC;

	float aFac = sphereA.r * rcpDistC;
	float bFac = -sphereB.r * rcpDistC;

	info.a_x = sphereA.x + aFac * distX;
	info.a_y = sphereA.y + aFac * distY;
	info.a_z = sphereA.z + aFac * distZ;

	info.b_x = sphereB.x + bFac * distX;
	info.b_y = sphereB.y + bFac * distY;
	info.b_z = sphereB.z + bFac * distZ;

//	collisionInfo[pos++] = info;
        pos += 1;
}

__global__ void computeCollisionInfo(Sphere *spheres, uint *sphereIDs,
		                             GPUBinData *binData, uint *objectCount,
		                             int *detectCount, uint *collisionPairs,
		                             CollisionInfo *collisionInfo,
		                             int numOfThreads, float halfBinRate) {
	int threadId = blockIdx.x * blockDim.x + threadIdx.x;
	uint *pairPtr = collisionPairs + (((threadId / 32) * 32) * MAX_OBJECTS_PER_BIN + (threadId % 32));

	if (threadId < numOfThreads) {
		uint startIndex = binData[threadId].startIndex;
		uint numOfObjects = objectCount[threadId];
		int  pos = (threadId == 0) ? 0 : detectCount[threadId-1];
		int  collisionCount = detectCount[threadId] - pos;

		uint localSphereIDs[MAX_OBJECTS_PER_BIN];
		Sphere localSpheres[MAX_OBJECTS_PER_BIN];
		for (uint i = 0; i < numOfObjects; i++) {
			uint ID = sphereIDs[startIndex+i];
			localSphereIDs[i] = ID;
			localSpheres[i] = spheres[ID];
		}

		for (int i = 0; i < collisionCount; i++) {
			uint pairs = *pairPtr;
			uint aIndex = pairs >> 16;
			uint bIndex = pairs & 0xFFFF;
			uint aID = localSphereIDs[aIndex];
			uint bID = localSphereIDs[bIndex];

			Sphere sphereA = localSpheres[aIndex];
			Sphere sphereB = localSpheres[bIndex];

			computePair(sphereA, sphereB,
					    aID, bID,
					    pos, collisionInfo);

			pairPtr += 32;
		}

	}
}


extern TimeRecorder<void> timer;

void binning(Sphere *_spheres, int numOfSpheres, float binSize,
			 int *_numOfBins, BinSphereBound *_bounds,
			 uint *_binIDs, uint *_sphereIDs, int &numOfPairs,
			 cudaStream_t &stream) {
	int threadsPerBlock		= 128;
	int maxNumOfBlocks		= 4096;
	int maxNumOfThreads		= maxNumOfBlocks * threadsPerBlock;
	int spheresPerThread	= MC_CeilDivide(numOfSpheres, maxNumOfThreads);
	int spheresPerBlock		= threadsPerBlock * spheresPerThread;
	int numOfBlocks = MC_CeilDivide(numOfSpheres, spheresPerBlock);

	float binRate = 1.0 / binSize;
	BinningStep1<<<numOfBlocks, threadsPerBlock, 0, stream>>>(
				_spheres, _bounds, _numOfBins,
				binRate, numOfSpheres, spheresPerThread);


	thrust::device_ptr<int> _numOfBinsArray(_numOfBins);
	int lastBinSize	= _numOfBinsArray[numOfSpheres - 1];

	thrust::exclusive_scan (_numOfBinsArray , _numOfBinsArray + numOfSpheres , _numOfBinsArray);
	numOfPairs	= _numOfBinsArray[numOfSpheres - 1] + lastBinSize;

	BinningStep2<<<numOfBlocks, threadsPerBlock, 0, stream>>>(
				_bounds, _numOfBins,
				_binIDs, _sphereIDs,
				numOfSpheres, spheresPerThread);

	thrust::device_ptr<uint> _binIDArray(_binIDs);
	thrust::device_ptr<uint> _sphereIDArray(_sphereIDs);
	thrust::stable_sort_by_key(_binIDArray, _binIDArray + numOfPairs, _sphereIDArray);
}

void createBinData(uint *_binIDs, uint *_sphereIDs, int numOfPairs,
				   uint *_startIndices, uint *_endIndices,
			       GPUBinData *_binData, uint *_objectCount, int &numOfBinData,
			       cudaStream_t &stream) {

	int threadsPerBlock		= 128;

	int maxNumOfBlocks		= 4096;
	int maxNumOfThreads		= maxNumOfBlocks * threadsPerBlock;
	int pairsPerThread		= MC_CeilDivide(numOfPairs, maxNumOfThreads);
	int pairsPerBlock		= threadsPerBlock * pairsPerThread;
	int numOfBlocks 		= MC_CeilDivide(numOfPairs, pairsPerBlock);

	BinDataStep1<<<numOfBlocks, threadsPerBlock, 0, stream>>>(
				_binIDs, _startIndices, _endIndices,
				numOfPairs, pairsPerThread);

	thrust::device_ptr<uint> _startIndexArray(_startIndices);
	thrust::device_ptr<uint> _endIndexArray(_endIndices);
	thrust::exclusive_scan (_startIndexArray, _startIndexArray + numOfPairs , _startIndexArray);
	thrust::exclusive_scan (_endIndexArray, _endIndexArray + numOfPairs , _endIndexArray);

	numOfBinData = _startIndexArray[numOfPairs-1];

	BinDataStep2<<<numOfBlocks, threadsPerBlock, 0, stream>>>(
				_binIDs, _startIndices, _binData,
				numOfPairs, pairsPerThread);

	BinDataStep3<<<numOfBlocks, threadsPerBlock, 0, stream>>>(
				_binIDs, _endIndices, _binData, _objectCount,
				numOfPairs, pairsPerThread);

	thrust::device_ptr<GPUBinData> _binDataArray(_binData);
	thrust::device_ptr<uint> _objectCountArray(_objectCount);
	thrust::stable_sort_by_key(_objectCountArray, _objectCountArray + numOfBinData, _binDataArray);
}

void gpu_stage7(Sphere *_spheres, uint *_sphereIDs, int numOfPairs,
				GPUBinData *_binData, uint *_objectCount, int numOfBinData,
				int *_detectCount, uint *_collisionPairs,
				CollisionInfo *&_collisionInfo, float binSize,
				cudaStream_t &stream) {
	float halfBinRate = 0.5 / binSize;
	int numOfThreads = numOfBinData;

	int threadsPerBlock = 64;
	int numOfBlocks = (numOfThreads + threadsPerBlock - 1) / threadsPerBlock;
	timer.appendStart("cd");	

	checkCollision<<<numOfBlocks, threadsPerBlock, 0, stream>>>(
				_spheres, _sphereIDs,
				_binData, _objectCount,
				_detectCount, _collisionPairs,
				numOfThreads, halfBinRate
	);

	thrust::device_ptr<int> _detectCountArray(_detectCount);
	thrust::inclusive_scan(_detectCountArray, _detectCountArray + numOfBinData, _detectCountArray);
	int numOfCollisions = _detectCountArray[numOfBinData - 1];

	printf("number of collisions = %d\n", numOfCollisions);

	timer.appendEnd("cd");
	timer.appendStart("info");

	CUDA_CHECK_MALLOC(_collisionInfo, CollisionInfo, numOfCollisions);

	threadsPerBlock = 64;
	numOfBlocks = (numOfThreads + threadsPerBlock - 1) / threadsPerBlock;

	computeCollisionInfo<<<numOfBlocks, threadsPerBlock>>>(
				_spheres, _sphereIDs,
				_binData, _objectCount,
				_detectCount, _collisionPairs,
				_collisionInfo,
				numOfThreads, halfBinRate
	);
	CUDA_CHECK_RETURN(cudaThreadSynchronize());	// Wait for the GPU launched work to complete
	CUDA_CHECK_RETURN(cudaGetLastError());

//	printf("kernel time = %lf\n", timer.getTotalTime("kernel"));
	CUDA_CHECK_FREE(_collisionInfo);

	timer.appendEnd("info");
}


struct CDThreadContext {
	cudaStream_t	stream;
	cudaEvent_t		event;

	Sphere			*_spheres;
	int				*_numOfBins;
	BinSphereBound	*_bounds;
	uint			*_binIDs;
	uint			*_sphereIDs;
	uint			*_startIndices;
	uint			*_endIndices;
	GPUBinData		*_binData;
	uint			*_objectCount;
	int				*_detectCount;

	uint			*_collisionPairs;
};

struct CDParameters {
	SphereArray *sphereArray;
	float binSize;
};

void CDThreadInit(int threadID);
void CDThread(int threadID, CDParameters *sphereArray);
void CDThreadFinalize(int threadID);

class CDWorker {
	CDThreadContext *threadContext;
	MultiThreadDelegator<CDParameters *> scheduler;

	friend void CDThreadInit(int threadID);
	friend void CDThread(int threadID, CDParameters *sphereArray);
	friend void CDThreadFinalize(int threadID);

	int maxSpheres;
	int maxPairs;
	int maxBins;
	int maxCollisions;

public:
	void init() {
		int deviceCount = 0;
		CUDA_CHECK_RETURN( cudaGetDeviceCount(&deviceCount) );
		printf("# of devices detected = %d\n", deviceCount);

//		deviceCount = 1;

		maxSpheres	= 1 * 1024 * 1024;
		maxPairs	= 32 * 1024 * 1024;
		maxBins		= 512 * 1024;
		maxCollisions = maxBins * 256;

		threadContext = new CDThreadContext[deviceCount];

		scheduler.init(deviceCount, CDThread,
					   CDThreadInit, CDThreadFinalize);
	}

	void schedule(SphereArray *sphereArray, float binSize) {
		CDParameters *para = new CDParameters;
		para->sphereArray = sphereArray;
		para->binSize = binSize;
		scheduler.schedule(para);
	}

	void finalize() {
		scheduler.finalize();
	}
};

CDWorker worker;

void CDThreadInit(int threadID) {
	CDThreadContext &ctx = worker.threadContext[threadID];

	cudaDeviceSetCacheConfig( cudaFuncCachePreferL1 );

	CUDA_CHECK_MALLOC(ctx._spheres,		Sphere,				worker.maxSpheres);
	CUDA_CHECK_MALLOC(ctx._bounds,		BinSphereBound,		worker.maxSpheres);
	CUDA_CHECK_MALLOC(ctx._numOfBins,	int,				worker.maxSpheres);
	CUDA_CHECK_MALLOC(ctx._binIDs,		uint,				worker.maxPairs);
	CUDA_CHECK_MALLOC(ctx._sphereIDs,	uint,				worker.maxPairs);
	CUDA_CHECK_MALLOC(ctx._startIndices,uint,				worker.maxPairs);
	CUDA_CHECK_MALLOC(ctx._endIndices,	uint,				worker.maxPairs);
	CUDA_CHECK_MALLOC(ctx._binData,		GPUBinData,			worker.maxBins);
	CUDA_CHECK_MALLOC(ctx._objectCount,	uint,				worker.maxBins);
	CUDA_CHECK_MALLOC(ctx._detectCount, int, 				worker.maxBins);

	CUDA_CHECK_MALLOC(ctx._collisionPairs,	uint,		worker.maxCollisions);
}

void CDThreadFinalize(int threadID) {
	CDThreadContext &ctx = worker.threadContext[threadID];

	CUDA_CHECK_FREE(ctx._spheres);
	CUDA_CHECK_FREE(ctx._bounds);
	CUDA_CHECK_FREE(ctx._numOfBins);
	CUDA_CHECK_FREE(ctx._binIDs);
	CUDA_CHECK_FREE(ctx._sphereIDs);
	CUDA_CHECK_FREE(ctx._startIndices);
	CUDA_CHECK_FREE(ctx._endIndices);
	CUDA_CHECK_FREE(ctx._binData);
	CUDA_CHECK_FREE(ctx._objectCount);
	CUDA_CHECK_FREE(ctx._detectCount);

	CUDA_CHECK_FREE(ctx._collisionPairs);
}


void CDThread(int threadID, CDParameters *para) {
	timer.appendStart("total");
	CDThreadContext &ctx = worker.threadContext[threadID];

	cudaStream_t &stream = ctx.stream;
//	cudaEvent_t &event = ctx.event;

	char thName[16];
	strcpy(thName, "Thread ");
	thName[7] = threadID + 48;
	thName[8] = 0;

	timer.appendStart("copy");

	SphereArray *sphereArray = para->sphereArray;
	float binSize = para->binSize;

	int numOfSpheres = sphereArray->size;
	Sphere *_spheres = ctx._spheres;
	CUDA_CHECK_COPY(_spheres, sphereArray->objects, Sphere, numOfSpheres, cudaMemcpyHostToDevice);

	timer.appendEnd("copy");

	timer.appendStart("binning");

	int numOfPairs;
	uint *_binIDs = ctx._binIDs;
	uint *_sphereIDs = ctx._sphereIDs;
	int *_numOfBins = ctx._numOfBins;
	BinSphereBound *_bounds = ctx._bounds;

//	timer.appendStart(thName);

	binning(_spheres, numOfSpheres, binSize,
			_numOfBins, _bounds,
			_binIDs, _sphereIDs, numOfPairs,
			stream);

//	timer.appendEnd(thName);

	timer.appendEnd("binning");

	timer.appendStart("prepare");

	int numOfBinData;
	GPUBinData *_binData = ctx._binData;
	uint *_objectCount = ctx._objectCount;
	uint *_startIndices = ctx._startIndices;
	uint *_endIndices = ctx._endIndices;

	createBinData(_binIDs, _sphereIDs, numOfPairs,
				  _startIndices, _endIndices,
				  _binData, _objectCount, numOfBinData,
				  stream);

	timer.appendEnd("prepare");

	int *_detectCount = ctx._detectCount;
	uint *_collisionPairs = ctx._collisionPairs;
	CollisionInfo *_collisionInfo;

	gpu_stage7(_spheres, _sphereIDs, numOfPairs,
			   _binData, _objectCount, numOfBinData,
			   _detectCount, _collisionPairs,
			   _collisionInfo, binSize,
			   stream);

	timer.appendEnd("total");
//	printf("total time: %lf\n", timer.getTotalTime("total"));
/*
	printf("numOfPairs = %d\n", numOfPairs);
	printf("number of bin data: %d\n", numOfBinData);

 	uint *objectCount = new uint[numOfBinData];
	CUDA_CHECK_COPY(objectCount, _objectCount, uint, numOfBinData, cudaMemcpyDeviceToHost);

	uint cdSum = 0;
	uint cdMax = 0;
	for (int i = 0; i < numOfBinData; i++) {
		uint n = objectCount[i];
		cdSum += n * n;
		if (n > cdMax) cdMax = n;
	}
	printf("total number of pairs processed: %u\n", cdSum);
	printf("max bin size: %u\n", cdMax);
	delete[] objectCount;

	printf("[%d] finish\n", threadID);
*/
}

#include <vector>
using namespace std;

void gpu_detect(SphereArray *sphereArray, float binSize) {
	worker.init();

/*	timer.appendStart("schedule");
	for (int i = 0; i < 10; i++) {
		worker.schedule(sphereArray, binSize);
	}
	timer.appendEnd("schedule");
	printf("overall time %lf\n", timer.getTotalTime("schedule"));
*/
	worker.schedule(sphereArray, binSize);

	worker.finalize();

	printf("copy time              = %lf\n", timer.getTotalTime("copy"));
	printf("binning time           =  %lf\n", timer.getTotalTime("binning"));
	printf("prepare time           = %lf\n", timer.getTotalTime("prepare"));
	printf("collision detect time  = %lf\n", timer.getTotalTime("cd"));
        printf("collision info time    = %lf\n", timer.getTotalTime("info"));
	printf("total time %lf\n", timer.getTotalTime("total"));

//	timer.printTimeLine("Thread 0");
//	timer.printTimeLine("Thread 1");
//	timer.printTimeLine("Time");
}
