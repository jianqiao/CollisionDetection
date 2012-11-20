CC = g++
CC_FLAGS = -O3 -funroll-loops -march=nocona -maccumulate-outgoing-args -Wno-unused-result -lm #-std=c99
LD_FLAGS = -L/usr/local/cuda/5.0/cuda/lib64 -lcuda -lcudart #-ltbb -ltbbmalloc_proxy -ltbbmalloc
CUDACC = nvcc
CUFLAGS = -O3 -cuda -arch=sm_20 --ptxas-options=-v

all: main cleanup

clean:
	rm -f main

cleanup:
	rm -f coldetect.cpp coldetect.o

coldetect.cpp: coldetect.h coldetect.cu Makefile
	$(CUDACC) $(CUFLAGS) coldetect.cu -o coldetect.cpp

main: main.C stages.h stages.C types.h utils.h coldetect.cpp Makefile
	$(CC) $(CC_FLAGS) $(LD_FLAGS) main.C stages.C coldetect.cpp -o main
	
	