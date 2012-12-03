#include <stdio.h>
#include "types.h"
#include "utils.h"

#include "stages.h"
#include "coldetect.h"

TimeRecorder<void> timer;

namespace para {
	int		xdim	= 100;
	int		ydim	= 100;
	int		zdim	= 100;

	float	radius	= 100;
	float	dist	= 150;
	float	binSize	= 300;
}

extern
int main(int argc, char *argv[]) {
	SphereArray *sphereArray = initSphereArray(para::xdim, para::ydim, para::zdim, para::radius, para::dist);
//	SphereArray *sphereArray = initRandomSphereArray(para::xdim, para::ydim, para::zdim, para::radius, para::dist);
//	rotate(sphereArray);

	gpu_detect(sphereArray, para::binSize);

/*
	timer.appendStart("CD");
	BinSpherePairArray *pairArray = stage1To3(sphereArray, para::binSize);
	stage4(pairArray);
	BinSphereDataArray *dataArray = stage5To6(pairArray);
	stage7(sphereArray, pairArray, dataArray, para::binSize);
	timer.appendEnd("CD");
*/

/*
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
	uint cdMax = 0;
	for (int i = 0; i < dataArray->size; i++) {
		uint n = dataArray->objects[i].numOfObjects;
		printf("%d\n", n);
		cdSum += n * n;
		if (n > cdMax) cdMax = n;
	}
	printf("total number of pairs processed: %u\n", cdSum);
	printf("max bin size: %u\n", cdMax);

	printf("total time for CD: %lf\n", timer.getTotalTime("CD"));
*/
	return 0;
}
