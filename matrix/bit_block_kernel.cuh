/*
 * bit_block_kernel.h
 *
 *  Created on: Dec 10, 2010
 *      Author: hans
 */

#ifndef BIT_BLOCK_KERNEL_H_
#define BIT_BLOCK_KERNEL_H_

#include <stdint.h>
#include "../util/xor.h"

namespace cuda_bwc{

__inline__ __device__ uint32_t rotr(uint32_t v,int n) {  return (v >> n) | (v << (32 - n) ); } 

template <typename ValueType, uint32_t THREADS_PER_BLOCK>
__global__ void
bitBlockKernel_32_64(const uint32_t num_rows,
				const uint32_t num_cols,
                const uint32_t * Aj,
                const ValueType *V,
                const ValueType * x,
                      ValueType * y)
{
    const uint32_t num_bits = 32;
    const uint32_t mask = num_bits - 1;
    const uint32_t sdataIndex = threadIdx.x & ~mask;
    const uint32_t gridSize = gridDim.x * THREADS_PER_BLOCK;
    const uint32_t numChunks = (num_rows + num_bits - 1) / num_bits;
    const uint32_t startCol = blockIdx.x * THREADS_PER_BLOCK + threadIdx.x;
    const uint32_t repeat = sizeof(ValueType)*8/num_bits;
    __shared__ volatile ValueType sdata[THREADS_PER_BLOCK];

    for(uint32_t chunk=0; chunk<numChunks; chunk++){
    	sdata[threadIdx.x] = 0;

		for(uint32_t col=startCol; col<num_cols; col+=gridSize){
			uint32_t bits = 0;
			ValueType val = 0;
			bits = Aj[chunk*num_cols+col];
			val = fetch_x(col, x);
            
            uint32_t r = threadIdx.x & mask;
            bits = rotr(bits, r);
            for(uint32_t i=0; i<num_bits; i++){
                if (bits & 1) {
                    sdata[sdataIndex + r] += val * V[32*(chunk*num_cols+col) + r];
                }
                bits >>=1;
                r = (r+1)&mask;
            }
            
			//uint32_t r = threadIdx.x & mask;
			//for(uint32_t i=0; i<num_bits; i++){
			//	uint32_t find_mask = 1<<r;
			//	if(find_mask & bits){
			//		sdata[sdataIndex + r] += val * V[32*(chunk*num_cols+col) + r];
			//	}
			//	r = (r+1) & mask;
			//}
		}

		syncthreads();

		for(uint32_t i=THREADS_PER_BLOCK/2; i>=num_bits; i>>=1){
			if(threadIdx.x<i){
				sdata[threadIdx.x] += sdata[threadIdx.x+i];
			}
//			syncthreads();
		}
        
        if (threadIdx.x < num_bits) {
            atomic_add( &y[chunk*num_bits + threadIdx.x] , sdata[threadIdx.x] );
        }
		
//		if(threadIdx.x < num_bits*repeat){
//			atomicXor(reinterpret_cast<uint32_t*>(&y[chunk*num_bits+threadIdx.x/repeat])+(threadIdx.x&(repeat-1)),
//						reinterpret_cast<volatile uint32_t*>(&sdata[threadIdx.x/repeat])[threadIdx.x&(repeat-1)]);
//		}
		
	
    }
}

template <typename ValueType, uint32_t THREADS_PER_BLOCK>
__global__ void
bitBlockKernel_128_256(const uint32_t num_rows,
				const uint32_t num_cols,
                const uint32_t * Aj,
                const ValueType * x,
                      ValueType * y)
{
    const uint32_t num_bits = 32;
    const uint32_t mask = num_bits - 1;
    const uint32_t sdataIndex = threadIdx.x & ~mask;
    const uint32_t gridSize = gridDim.x * THREADS_PER_BLOCK;
    const uint32_t numChunks = (num_rows + num_bits - 1) / num_bits;
    const uint32_t startCol = blockIdx.x * THREADS_PER_BLOCK + threadIdx.x;
    __shared__ volatile uint64_t sdata[sizeof(ValueType)/sizeof(uint64_t)][THREADS_PER_BLOCK];

    for(uint32_t chunk=0; chunk<numChunks; chunk++){
    	scattered_write<sizeof(ValueType)/sizeof(uint64_t)>(&sdata[0][threadIdx.x], THREADS_PER_BLOCK, 0L);

		for(uint32_t col=startCol; col<num_cols; col+=gridSize){
			uint32_t bits = 0;
			ValueType val = 0;
			bits = Aj[chunk*num_cols+col];
			val = fetch_x(col, x);
			uint32_t r = threadIdx.x & mask;
			for(uint32_t i=0; i<num_bits; i++){
				uint32_t find_mask = 1<<r;
				if(find_mask & bits){
					XOR(&sdata[0][sdataIndex + r],THREADS_PER_BLOCK, val);
				}
				r = (r+1) & mask;
			}
		}

		syncthreads();

		for(uint32_t i=THREADS_PER_BLOCK/2; i>=num_bits; i/=2){
			if(threadIdx.x<i){
				XOR<sizeof(ValueType)/sizeof(uint64_t)>(&sdata[0][threadIdx.x], &sdata[0][threadIdx.x+i], THREADS_PER_BLOCK);
			}
			syncthreads();
		}

		if(threadIdx.x < num_bits){
			atomic_XOR<sizeof(ValueType)/sizeof(uint64_t)>(reinterpret_cast<uint32_t*>(&y[chunk*num_bits+threadIdx.x]),
					reinterpret_cast<volatile uint32_t*>(&sdata[0][threadIdx.x]), THREADS_PER_BLOCK*2);
		}
		syncthreads();
    }
}
}
#endif /* BIT_BLOCK_KERNEL_H_ */
