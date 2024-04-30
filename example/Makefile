CXX = g++
CXXFLAGS = -std=c++14 -O3 -Wall -Wextra -fopenmp
INC_DIR = ../RAJA-install/include
LIB_DIR = ../RAJA-install/lib
LIBS = -lRAJA -lm

TARGETS: serial raja openmp pthread

all: serial cuda pthread raja openmp

cuda: cuda.cu
	nvcc $(NFLAGS) -o out/$@ $< $(LIB)

pthread: pthread.cpp
	$(CXX) $(CXXFLAGS) -o out/$@ $< $(LIB) -lpthread

raja: raja.cpp
	$(CXX) $(CXXFLAGS) -I$(INC_DIR) -o out/$@ $^ -L$(LIB_DIR) $(LIBS)

serial: serial.cpp
	$(CXX) $(CXXFLAGS) -o out/$@ $< $(LIB)

openmp: openmp.cpp
	$(CXX) $(CXXFLAGS) -o out/$@ $< $(LIB) -fopenmp

.PHONY: clean

clean:
	rm -f out/*
