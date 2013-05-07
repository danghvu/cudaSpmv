COMMON = -O2 -arch=sm_20 --ptxas-options="--maxrregcount=32" matrix/MatrixInput.cpp parameters.cpp params_parser.cpp
CUSP = ../cusplibrary

all: spmv prep

spmv: main.cu Makefile matrix/MatrixInput.cpp
	nvcc $(COMMON) main.cu -o spmv -I$(CUSP)

prep: Makefile prep.cu matrix/MatrixInput.cpp CutOffFinder.h
	nvcc $(COMMON) prep.cu BalancedMatrix.cpp -o prep -I$(CUSP)

clean:
	rm spmv; rm prep;

