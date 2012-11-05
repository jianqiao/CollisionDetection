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
#include <vector>
#include <map>
#include <string>

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

inline void resetTimer(hwtimer_t* timer)
{
	hrtime_t start = 0;
	hrtime_t end = 0;
}

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

	resetTimer(timer);
}

inline void startTimer(hwtimer_t* timer)
{
	timer->start = _rdtsc();
}

inline void stopTimer(hwtimer_t* timer)
{
	timer->end = _rdtsc();
}

inline uint64_t getTimerTicks(hwtimer_t* timer)
{
	return timer->end - timer->start;
}

inline uint64_t getTimerNs(hwtimer_t* timer)
{
	if (timer->cpuMHz == 0) {
		/* Cannot use a timer without first initializing it
		   or if not on linux
		*/
		return 0;
	}
	return (uint64_t)(((double)getTimerTicks(timer))/timer->cpuMHz*1000);
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


#endif /* UTILS_H_ */
