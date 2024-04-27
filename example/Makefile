CXX = g++
CXXFLAGS = -std=c++14 -O3 -Wall -Wextra -fopenmp
INC_DIR = ../RAJA-install/include
LIB_DIR = ../RAJA-install/lib
LIBS = -lRAJA -lm

TARGETS: serial raja cuda openmp pthread

all: $(TARGETS)

cuda: cuda.cu
	nvcc $(NFLAGS) -o $@ $< $(LIB)

pthread: pthread.cpp
	$(CXX) $(CXXFLAGS) -o $@ $< $(LIB) -lpthread

raja: raja.cpp
	$(CXX) $(CXXFLAGS) -I$(INC_DIR) -o $@ $^ -L$(LIB_DIR) $(LIBS)

serial: serial.cpp
	$(CXX) $(CXXFLAGS) -o $@ $< $(LIB)

openmp: openmp.cpp
	$(CXX) $(CXXFLAGS) -o $@ $< $(LIB) -fopenmp

.PHONY: clean

clean:
	rm -f $(TARGETS)
