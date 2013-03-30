COMMON = -O2 -arch=sm_30 --ptxas-options="" matrix/MatrixInput.cpp parameters.cpp params_parser.cpp

all: spmv prep

spmv: main.cu Makefile matrix/MatrixInput.cpp
	nvcc $(COMMON) main.cu -o spmv -I/home/danghvu/workspace/cusp-library

prep: Makefile prep.cu matrix/MatrixInput.cpp CutOffFinder.h
	nvcc $(COMMON) prep.cu BalancedMatrix.cpp -o prep -I/home/danghvu/workspace/cusp-library

clean:
	rm spmv; rm prep;

