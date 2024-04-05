CC=gcc
CFLAGS=-g -Wall -Wno-unknown-pragmas -O3
NFLAGS=-ccbin $(CC) -Xcompiler "$(CFLAGS)"
LIB=-lm
TARGETS=gaus_cuda

all: $(TARGETS)

gaus_cuda: gaus_cuda.cu
	nvcc $(NFLAGS) -o $@ $< $(LIB)

clean:
	rm -f $(TARGETS)
