#ifndef STAGES_H_
#define STAGES_H_

#include "types.h"

/* initialize the sphere array */
SphereArray*
	initSphereArray(int xdim, int ydim, int zdim, float radius, float dist);

SphereArray*
	initRandomSphereArray(int xdim, int ydim, int zdim, float radius, float dist);

void
	rotate(SphereArray *sphereArray);

/* binning */
BinSpherePairArray*
	stage1To3(SphereArray* sphereArray, float binSize);

/* radix sort */
void
	stage4(BinSpherePairArray* pairArray);

/* radix sort bins according to # of objects inside */
BinSphereDataArray*
	stage5To6(BinSpherePairArray *pairArray);

void
	stage7(SphereArray* sphereArray, BinSpherePairArray* pairArray, BinSphereDataArray *dataArray, float binSize);


#endif /* STAGES_H_ */
