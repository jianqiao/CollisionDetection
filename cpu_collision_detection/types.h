/*
 * types.h
 *
 *  Created on: Nov 2, 2012
 *      Author: jianqiao
 */

#ifndef TYPES_H_
#define TYPES_H_

typedef unsigned int uint;
typedef unsigned long long ull;

struct Sphere {
	float x, y, z, r;
};

template <class T>
struct SimpleArray {
	uint	size;
	T*		objects;
};

template <class T>
struct BinObjectPair {
	uint	binID;
	uint 	objectID;
};

typedef BinObjectPair<Sphere>		BinSpherePair;

template <class T>
struct BinData {
	uint	binID;
	uint	startIndex;
	uint	numOfObjects;
};
typedef BinData<Sphere>				BinSphereData;

typedef SimpleArray<Sphere>			SphereArray;
typedef SimpleArray<BinSpherePair>	BinSpherePairArray;
typedef SimpleArray<BinSphereData>	BinSphereDataArray;

#endif /* TYPES_H_ */
