/*
 * main.C
 *
 *  Created on: Nov 2, 2012
 *      Author: jianqiao
 */

#include <stdio.h>
#include "types.h"
#include "utils.h"

#include "stages.h"
#include "coldetect.h"

TimeRecorder<void> timer;

namespace para {
	int		xdim	= 20;
	int		ydim	= 20;
	int		zdim	= 20;

	float	radius	= 100;
	float	dist	= 160;

	float	binSize	= 273;
}


int main(int argc, char *argv[]) {
	SphereArray *sphereArray = initSphereArray(para::xdim, para::ydim, para::zdim, para::radius, para::dist);

	timer.appendStart("binning");
	BinSpherePairArray *pairArray = stage1To3(sphereArray, para::binSize);
	timer.appendEnd("binning");

	timer.appendStart("sorting");
	stage4(pairArray);
	timer.appendEnd("sorting");

	timer.appendStart("data");
	BinSphereDataArray *dataArray = stage5To6(pairArray);
	timer.appendEnd("data");

	timer.appendStart("CD");
//	stage7(sphereArray, pairArray, dataArray, para::binSize);
	gpu_stage7(sphereArray, pairArray, dataArray, para::binSize);
	timer.appendEnd("CD");

	printf("number of spheres = %u, memory = %lf M\n",
			sphereArray->size,
			(sphereArray->size * sizeof(Sphere)) / 1024.0 / 1024.0);
	printf("number of bin-sphere pairs (array B) = %u, memory = %lf M\n",
			pairArray->size,
			(pairArray->size * sizeof(BinSpherePair)) / 1024.0 / 1024.0);
	printf("number of bin data (array C) = %u, memory = %lf M\n",
			dataArray->size,
			(dataArray->size * sizeof(BinSphereData)) / 1024.0 / 1024.0);

	uint cdSum = 0;
	for (int i = 0; i < dataArray->size; i++) {
		uint n = dataArray->objects[i].numOfObjects;
		cdSum += n * n;
	}
	printf("total number of pairs processed: %u\n", cdSum);

	printf("total time for binning: %lf\n", timer.getTotalTime("binning"));
	printf("total time for sorting: %lf\n", timer.getTotalTime("sorting"));
	printf("total time for rearranging bins: %lf\n", timer.getTotalTime("data"));
	printf("total time for CD: %lf\n", timer.getTotalTime("CD"));

	return 0;
}
