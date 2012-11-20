/*
 * coldetect.h
 *
 *  Created on: Nov 5, 2012
 *      Author: jianqiao
 */

#ifndef COLDETECT_H_
#define COLDETECT_H_

#include "types.h"

void gpu_stage7(SphereArray* sphereArray, BinSpherePairArray* pairArray, BinSphereDataArray *dataArray, float binSize);

BinSpherePairArray* gpu_stage1To4(SphereArray* sphereArray, float binSize);

#endif /* COLDETECT_H_ */
