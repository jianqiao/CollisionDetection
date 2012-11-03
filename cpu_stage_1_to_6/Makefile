CC = g++
CC_FLAGS = #-std=c99
LD_FLAGS = #-ltbb -ltbbmalloc_proxy -ltbbmalloc
#DEBUG = 1

ifeq ($(DEBUG), 1)
	CC_FLAGS += -g
else
	CC_FLAGS += -O2
endif

all: main

clean:
	rm -f main

main: main.C stages.h stages.C types.h utils.h
	$(CC) $(CC_FLAGS) $(LD_FLAGS) main.C stages.C -o main
	
