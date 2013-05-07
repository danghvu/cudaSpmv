# vim: noexpandtab 
CUDA_PATH = /usr/local/cuda
NVCC = $(CUDA_PATH)/bin/nvcc
CUDA_INCLUDE = $(CUDA_PATH)/include
CUSP = ../cusplibrary

CPP_FILES := $(shell find $(SOURCEDIR) -name '*.cpp')
OBJ_FILES := $(addprefix obj/,$(CPP_FILES:.cpp=.o))

CC_FLAGS = -O2 -I$(CUDA_INCLUDE) -I$(CUSP)

COMMON = -O2 -arch=sm_20 --ptxas-options="--maxrregcount=32" $(OBJ_FILES) -I$(CUDA_INCLUDE) -I$(CUSP)

all: spmv prep

spmv: main.cu Makefile $(OBJ_FILES)
	$(NVCC) $(COMMON) main.cu -o spmv

prep: prep.cu $(OBJ_FILES)
	$(NVCC) $(COMMON) prep.cu -o prep

cusp: cusp.cu
	$(NVCC) -O2 -arch=sm_20 cusp.cu -o cusp -I$(CUSP)

obj/%.o: %.cpp
	@mkdir -p $(@D)
	g++ $(CC_FLAGS) -c -o $@ $<

clean:
	rm -rf spmv prep cusp test.mtx* obj/*


