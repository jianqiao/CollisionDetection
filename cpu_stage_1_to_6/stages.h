/*
 * stages.h
 *
 *  Created on: Nov 2, 2012
 *      Author: jianqiao
 */

#ifndef STAGES_H_
#define STAGES_H_

#include "types.h"

/* initialize the sphere array */
SphereArray*
	initSphereArray(int xdim, int ydim, int zdim, float radius, float dist);

/* binning */
BinSpherePairArray*
	stage1To3(SphereArray* sphereArray, float binSize);

/* radix sort */
void
	stage4(BinSpherePairArray* pairArray);

/* radix sort bins according to # of objects inside */
BinSphereDataArray*
	stage5(BinSpherePairArray *pairArray);




#endif /* STAGES_H_ */
