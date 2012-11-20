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


struct BinSphereBound {
	int minIDX, minIDY, minIDZ;
	int maxIDX, maxIDY, maxIDZ;
};


#define widthX	10
#define widthY	10
#define widthZ	10

#define maskX	((1 << widthX) - 1)
#define maskY	((1 << widthY) - 1)
#define maskZ	((1 << widthZ) - 1)

#define shiftX	(widthY + widthZ)
#define shiftY	(widthZ)

#define MC_Key(x,y,z) (((((uint)(x)) & maskX) << shiftX) | \
					   ((((uint)(y)) & maskY) << shiftY) | \
					    (((uint)(z)) & maskZ))


#endif /* TYPES_H_ */
