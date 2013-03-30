#ifndef SLICED_COO_KERNEL_H_
#define SLICED_COO_KERNEL_H_

#include <stdint.h>
#include <stdio.h>
#include "../util/util.h"
#include "../util/textures.h"
#include "../util/xor.h"

template <typename ValueType, const uint32_t THREADS_PER_BLOCK, const uint32_t NUM_ROWS_PER_SLICE, const uint32_t LANE_SIZE>
__global__ void
sliced_coo_kernel_32(
				const uint32_t num_rows,
                const uint32_t numPacks,
                const uint32_t * cols,
                const uint16_t * rows, 
                const ValueType * V,
                const uint32_t * offsets,
                const ValueType * x,
                      ValueType * y)
{
    const int thread_lane = threadIdx.x & (LANE_SIZE-1);
    const int row_lane = threadIdx.x/(LANE_SIZE);
 
    __shared__ ValueType sdata[NUM_ROWS_PER_SLICE][LANE_SIZE];
    
    const uint32_t packNo=blockIdx.x;
	const uint32_t limit = ( (packNo==numPacks-1)?((num_rows-1)%NUM_ROWS_PER_SLICE)+1:NUM_ROWS_PER_SLICE );

    const uint32_t begin = offsets[packNo];
    const uint32_t end = offsets[packNo+1];
    for(int i=row_lane; i<limit; i+=THREADS_PER_BLOCK/LANE_SIZE){
        sdata[i][thread_lane] = 0;
    }
    
    __syncthreads();

    for(int32_t index=begin+threadIdx.x; index<end; index+=THREADS_PER_BLOCK){
        const uint32_t col = cols[index];
        const uint16_t row = rows[index];
        const ValueType value = V[index];

        const ValueType input = fetch_x(col, x) * value;
        atomic_add(&sdata[row][thread_lane], input);
    }

    __syncthreads();

    for (uint32_t i=row_lane; i<limit; i+=THREADS_PER_BLOCK/LANE_SIZE) { 
        volatile ValueType *psdata = sdata[i];
        int tid = (thread_lane+i) & (LANE_SIZE - 1);

        if (LANE_SIZE>128 && thread_lane<128) psdata[tid]+=psdata[(tid+128) & (LANE_SIZE-1)]; __syncthreads();
        if (LANE_SIZE>64 && thread_lane<64) psdata[tid]+=psdata[(tid+64) & (LANE_SIZE-1)]; __syncthreads();
        if (LANE_SIZE>32 && thread_lane<32) psdata[tid]+=psdata[(tid+32) & (LANE_SIZE-1)]; __syncthreads();
        
        if (LANE_SIZE>16 && thread_lane<16) psdata[tid]+=psdata[( tid+16 ) & (LANE_SIZE-1)];
        if (LANE_SIZE>8 && thread_lane<8) psdata[tid]+=psdata[( tid+8 ) & (LANE_SIZE-1)];
        if (LANE_SIZE>4 && thread_lane<4) psdata[tid]+=psdata[( tid+4 ) & (LANE_SIZE-1)];
        if (LANE_SIZE>2 && thread_lane<2) psdata[tid]+=psdata[( tid+2 ) & (LANE_SIZE-1)];
        if (LANE_SIZE>1 && thread_lane<1) psdata[tid]+=psdata[( tid+1 ) & (LANE_SIZE-1)];
    }

    __syncthreads();
    const uint32_t actualRow = packNo * NUM_ROWS_PER_SLICE;

    for(int r = threadIdx.x; r < limit; r+=THREADS_PER_BLOCK){
        y[actualRow+r] = sdata[r][thread_lane];
    }
}
#endif /* SLICED_COO_KERNEL_H_ */
