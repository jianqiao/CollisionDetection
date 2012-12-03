/*
 * utils.h
 *
 *  Created on: Nov 2, 2012
 *      Author: jianqiao
 */

#ifndef UTILS_H_
#define UTILS_H_

#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <sys/time.h>
#include <pthread.h>
#include <semaphore.h>
#include <vector>
#include <map>
#include <string>
#include <stack>
using namespace std;

/////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////  Radix sort  /////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////

#define MCInner_RadixSort_32_16_OneRound(arr, item, auxArr, length, shift) { \
	memset(count, 0, 65536 * sizeof(unsigned int)); \
	for(unsigned int i = 0; i < length; i++) \
		++count[(arr[i].item >> shift) & MASK]; \
	position[0] = 0; \
	for(unsigned int i = 1; i < 65536; i++) \
		position[i] = position[i-1] + count[i-1]; \
	for(unsigned int i = 0; i < length; i++) \
		auxArr[position[(arr[i].item >> shift) & MASK]++] = arr[i]; \
}
#define MC_RadixSort_32_16(arr, item, auxArr, length) { \
	unsigned int count[65536]; \
	unsigned int position[65536]; \
	unsigned int MASK = 0xFFFF; \
	MCInner_RadixSort_32_16_OneRound(arr, item, auxArr, length, 0) \
	MCInner_RadixSort_32_16_OneRound(auxArr, item, arr, length, 16) \
}

/////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////  Other   ///////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////

#define MC_CeilDivide(x, y) \
	((x) + (y) - 1) / (y)

/////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////  Timer  ///////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////

typedef uint64_t hrtime_t;

__inline__ hrtime_t _rdtsc() {
    unsigned long int lo, hi;
    __asm__ __volatile__ ("rdtsc" : "=a" (lo), "=d" (hi));
    return (hrtime_t)hi << 32 | lo;
}

typedef struct {
	hrtime_t start;
	hrtime_t end;
	double cpuMHz;
} hwtimer_t;

inline void initTimer(hwtimer_t* timer)
{
#if defined(__linux) || defined(__linux__) || defined(linux)
    FILE* cpuinfo;
    char str[100];
    cpuinfo = fopen("/proc/cpuinfo","r");
    while(fgets(str,100,cpuinfo) != NULL){
        char cmp_str[8];
        strncpy(cmp_str, str, 7);
        cmp_str[7] = '\0';
        if (strcmp(cmp_str, "cpu MHz") == 0) {
			double cpu_mhz;
			sscanf(str, "cpu MHz : %lf", &cpu_mhz);
			timer->cpuMHz = cpu_mhz;
			break;
        }
    }
    fclose( cpuinfo );
#else
    timer->cpuMHz = 0;
#endif
}

template <class T>
class TimeRecorder {
private:
	int64_t referenceTime;
	int64_t timeUnit;
	struct TimeLine {
		int state;
		int length;
		std::vector<double> startTime;
		std::vector<double> endTime;
		TimeLine() {
			state = 0;
			length = 0;
		}
	};
	std::map<std::string, TimeLine> recorder;

private:
    double getElapsedTime(int64_t rtime) {
    	return ((double)(_rdtsc() - rtime)) / timeUnit;
    }

public:
	TimeRecorder() {
	    hwtimer_t timerInfo;
	    initTimer(&timerInfo);
	    timeUnit = (int64_t) (timerInfo.cpuMHz * 1000000);
		reset();
	}

	void reset() {
		referenceTime = _rdtsc();
	}

	void appendStart(std::string label = "default") {
		TimeLine& timeLine = recorder[label];
		if (timeLine.state == 1)
			return;
		timeLine.startTime.push_back(getElapsedTime(referenceTime));
		timeLine.state = 1;
	}

	void appendEnd(std::string label = "default") {
		TimeLine& timeLine = recorder[label];
		if (timeLine.state == 0)
			return;
		timeLine.endTime.push_back(getElapsedTime(referenceTime));
		timeLine.length += 1;
		timeLine.state = 0;
	}

	double getTotalTime(std::string label = "default") {
		TimeLine& timeLine = recorder[label];
		double totalTime = 0;
		for (int i = 0; i < timeLine.length; i++) {
			totalTime += timeLine.endTime[i] - timeLine.startTime[i];
		}
		return totalTime;
	}

	void printTimeLine(std::string label = "default", FILE *output = stdout) {
		TimeLine& timeLine = recorder[label];
		fprintf(output, "Label: %s  --  Total %d records.\n", label.c_str(), timeLine.length);
		for (unsigned int i = 0; i < timeLine.length; i++) {
			fprintf(output, "[%u] %lf -> %lf\n", i, timeLine.startTime[i], timeLine.endTime[i]);
		}
	}
};

template <class T>
void *MultiThreadDelegator_call_func(void *classRef);

template <class ArgType>
class MultiThreadDelegator {
private:
	typedef stack<int> ThreadPool;
	int numOfThreads;
	ArgType *argList;
	pthread_t *threadHandles;
	sem_t *threadSems;
	ThreadPool* threadPool;
	int *threadFlags;
	sem_t availableSem, finishSem;
	pthread_mutex_t occupy_mutex, finish_mutex;
	int createThreadCnt;
	int finishFlag;
	void (*funcWrapper)(int, ArgType);
	void (*threadInit)(int);
	void (*threadFinalize)(int);
private:
	void clear();
	int allocThread();
	void releaseThread(int threadId);
	int checkFinish(int threadId);
protected:
	void thread_run_func();
	template <class T>
	friend void *MultiThreadDelegator_call_func(void *classRef);
public:
	MultiThreadDelegator();
	~MultiThreadDelegator();

	void init(int numOfThreads, void (*threadRunFunc)(int, ArgType),
			  void (*threadInitFunc)(int) = NULL, void (*threadFinalizeFunc)(int) = NULL);
	int schedule(ArgType arg);
	void finalize();
};

template <class T>
void *MultiThreadDelegator_call_func(void *classRef) {
	MultiThreadDelegator<T> *delegator = (MultiThreadDelegator<T> *)classRef;
	delegator -> thread_run_func();
	return NULL;
}
template <class ArgType>
MultiThreadDelegator<ArgType>::MultiThreadDelegator() {
	threadHandles = NULL;
}
template <class ArgType>
MultiThreadDelegator<ArgType>::~MultiThreadDelegator() {
	clear();
}
template <class ArgType>
void MultiThreadDelegator<ArgType>::clear() {
	if (threadHandles != NULL) {
		delete[] threadHandles;
		delete[] threadSems;
		delete threadPool;
		delete[] threadFlags;
		delete[] argList;
	}
	threadHandles = NULL;
}
template <class ArgType>
int MultiThreadDelegator<ArgType>::allocThread() {
	pthread_mutex_lock(&occupy_mutex);
	int threadId = threadPool->top();
	threadPool->pop();
	threadFlags[threadId] = 1;
	pthread_mutex_unlock(&occupy_mutex);
	return threadId;
}
template <class ArgType>
void MultiThreadDelegator<ArgType>::releaseThread(int threadId) {
	pthread_mutex_lock(&occupy_mutex);
	threadPool->push(threadId);
	threadFlags[threadId] = 0;
	pthread_mutex_unlock(&occupy_mutex);
}
template <class ArgType>
void MultiThreadDelegator<ArgType>::init(int num, void (*inFuncWrapper)(int, ArgType),
	 	                                 void (*inThreadInit)(int),
	 	                                 void (*inThreadFinalize)(int)) {
	clear();
	numOfThreads = num;
	threadHandles = new pthread_t[numOfThreads];
	threadSems = new sem_t[numOfThreads];
	threadPool = new ThreadPool;
	threadFlags = new int[numOfThreads];
	argList = new ArgType[numOfThreads];
    funcWrapper = inFuncWrapper;
    threadInit = inThreadInit;
    threadFinalize = inThreadFinalize;
    finishFlag = 0;
    for (int i = 0; i < numOfThreads; i++) {
        sem_init(threadSems + i, 0, 0);
        threadFlags[i] = 0; // empty
    }
    sem_init(&availableSem, 0, 0);
    sem_init(&finishSem, 0, 0);
    pthread_mutex_init(&occupy_mutex, NULL);
    pthread_mutex_init(&finish_mutex, NULL);
    createThreadCnt = 0;
    for (int i = 0; i < numOfThreads; i++) {
        pthread_create(threadHandles + i, NULL, MultiThreadDelegator_call_func<ArgType>, this);
    }
}
template <class ArgType>
int MultiThreadDelegator<ArgType>::checkFinish(int threadId) {
	int flag;
	pthread_mutex_lock(&occupy_mutex);
	flag = finishFlag && (threadFlags[threadId] == 0);
	pthread_mutex_unlock(&occupy_mutex);
	return flag;
}
template <class ArgType>
void MultiThreadDelegator<ArgType>::thread_run_func() {
	pthread_mutex_lock(&occupy_mutex);
	int threadId = createThreadCnt++;
	pthread_mutex_unlock(&occupy_mutex);
	if (threadInit != NULL)
		threadInit(threadId);
	releaseThread(threadId);
	sem_post(&availableSem);
	sem_t* pThreadSem = threadSems + threadId;
	sem_wait(pThreadSem);
	while(checkFinish(threadId) == 0) {
		funcWrapper(threadId, argList[threadId]);
		releaseThread(threadId);
		sem_post(&availableSem);
		sem_wait(pThreadSem);
	}
	if (threadFinalize != NULL)
		threadFinalize(threadId);
}
template <class ArgType>
int MultiThreadDelegator<ArgType>::schedule(ArgType arg) {
	sem_wait(&availableSem);
	int threadId = allocThread();
	argList[threadId] = arg;
	sem_post(threadSems + threadId);
	return threadId;
}
template <class ArgType>
void MultiThreadDelegator<ArgType>::finalize() {
	pthread_mutex_lock(&occupy_mutex);
	finishFlag = 1;
	pthread_mutex_unlock(&occupy_mutex);
	for (int i = 0; i < numOfThreads; i++)
		sem_post(threadSems + i);
	for (int i = 0; i < numOfThreads; i++)
		pthread_join(threadHandles[i], NULL);
	clear();
}


#endif /* UTILS_H_ */
