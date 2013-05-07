/*
 * xor.h
 *
 *  Created on: Dec 14, 2010
 *      Author: hans
 */

#ifndef XOR_H_
#define XOR_H_

#include "../datatype/int128.h"
#include "../datatype/int256.h"

__device__ __inline__ static void gather(const uint32_t* mem, uint64_t *input, uint32_t stride){
    uint64_t temp = 0;
    temp = *mem;
    temp |= (static_cast<uint64_t>(*(mem+stride)) << 32);
    *input = temp;
}

__device__ __inline__ static void gather(const uint32_t* mem, Int128 *input, uint32_t stride){
    Int128 temp = 0;
    temp.m.x = *mem;
    temp.m.x |= (static_cast<uint64_t>(*(mem+stride)) << 32);
    temp.m.y |= (static_cast<uint64_t>(*(mem+stride*2)));
    temp.m.y |= (static_cast<uint64_t>(*(mem+stride*3)) << 32);

    *input = temp;
}

__device__ __inline__ static void gather(const uint32_t* mem, Int256 *input, uint32_t stride){
    Int256 temp = 0;
    temp.m.x = *mem;
    temp.m.x |= (static_cast<uint64_t>(*(mem+stride)) << 32);
    temp.m.y |= (static_cast<uint64_t>(*(mem+stride*2)));
    temp.m.y |= (static_cast<uint64_t>(*(mem+stride*3)) << 32);
    temp.m.z |= (static_cast<uint64_t>(*(mem+stride*4)));
    temp.m.z |= (static_cast<uint64_t>(*(mem+stride*5)) << 32);
    temp.m.w |= (static_cast<uint64_t>(*(mem+stride*6)));
    temp.m.w |= (static_cast<uint64_t>(*(mem+stride*7)) << 32);

    *input = temp;
}

template <class T, int REPEAT>
__device__ inline static void scattered_write(T * ar, uint32_t stride, T input){
    uint32_t dist = 0;
    for(int i=0; i<REPEAT-1; i++){
        ar[dist] = 0;
        dist += stride;
    }
    ar[dist] = input;
}

template <int REPEAT>
__device__ inline static void scattered_write(volatile uint64_t * ar, unsigned int stride, uint64_t input){
    unsigned int dist = 0;
    for(int i=0; i<REPEAT-1; i++){
        ar[dist] = 0;
        dist += stride;
    }
    ar[dist] = input;
}

__device__ __inline__ void atomic_add(float *address, float val) {
    atomicAdd(address, val);
}

/*
   __device__ __inline__ void atomic_add(double *address, double value)  //See CUDA official forum
   {
   unsigned long long oldval, newval, readback;

   oldval = __double_as_longlong(*address);
   newval = __double_as_longlong(__longlong_as_double(oldval) + value);
   while ((readback=atomicCAS((unsigned long long *)address, oldval, newval)) != oldval)
   {
   oldval = readback;
   newval = __double_as_longlong(__longlong_as_double(oldval) + value);
   }
   }
 */

__device__ __inline__ void atomic_add(double* address, double val) { 
    unsigned long long int* address_as_ull = (unsigned long long int*)address; 
    unsigned long long int old = *address_as_ull, assumed; 

    do { 
        assumed = old; 
        old = atomicCAS(address_as_ull, assumed, 
                __double_as_longlong( val +  __longlong_as_double(assumed)  ) 
                ); 

    } while (assumed != old); 
    //    return __longlong_as_double(old);
}

__device__ __inline__ void atomic_xor(uint32_t* data, uint32_t value){
    atomicXor(data, value);
}

__device__ __inline__ void atomic_xor(uint64_t* data, uint64_t value){
    atomicXor(reinterpret_cast<uint32_t*>(data), static_cast<uint32_t>(value));
    atomicXor(reinterpret_cast<uint32_t*>(data)+1, static_cast<uint32_t>(value>>32));
}

__device__ __inline__ void atomic_xor(Int128* data, const Int128& value) {
    atomicXor(reinterpret_cast<uint32_t*>(data), static_cast<uint32_t> (value.m.x));
    atomicXor(reinterpret_cast<uint32_t*>(data)+1, static_cast<uint32_t> (value.m.x >> 32));
    atomicXor(reinterpret_cast<uint32_t*>(data)+2, static_cast<uint32_t> (value.m.y));
    atomicXor(reinterpret_cast<uint32_t*>(data)+3, static_cast<uint32_t> (value.m.y >> 32));
}

__device__ __inline__ void atomic_xor(Int256* data, const Int256& value) {
    atomicXor(reinterpret_cast<uint32_t*>(data), static_cast<uint32_t> (value.m.x));
    atomicXor(reinterpret_cast<uint32_t*>(data)+1, static_cast<uint32_t> (value.m.x >> 32));
    atomicXor(reinterpret_cast<uint32_t*>(data)+2, static_cast<uint32_t> (value.m.y));
    atomicXor(reinterpret_cast<uint32_t*>(data)+3, static_cast<uint32_t> (value.m.y >> 32));
    atomicXor(reinterpret_cast<uint32_t*>(data)+4, static_cast<uint32_t> (value.m.z));
    atomicXor(reinterpret_cast<uint32_t*>(data)+5, static_cast<uint32_t> (value.m.z >> 32));
    atomicXor(reinterpret_cast<uint32_t*>(data)+6, static_cast<uint32_t> (value.m.w));
    atomicXor(reinterpret_cast<uint32_t*>(data)+7, static_cast<uint32_t> (value.m.w >> 32));
}

__device__ __inline__ void XOR(uint64_t* ar, uint32_t stride, const uint64_t& input){
    *ar ^= input;
}


__device__ __inline__ void XOR(uint64_t* ar, uint32_t stride, const Int128& input){
    ar[0] ^= input.m.x;
    ar[stride] ^= input.m.y;
}

__device__ __inline__ void XOR(uint64_t* ar, uint32_t stride, const Int256& input){
    ar[0] ^= input.m.x;
    ar[stride] ^= input.m.y;
    ar[stride*2] ^= input.m.z;
    ar[stride*3] ^= input.m.w;
}

template <int REPEAT>
__device__ __inline__ void XOR(uint64_t* ar, uint64_t* ar2, uint32_t stride){
    uint32_t dist = 0;
    for(int i=0; i<REPEAT; i++){
        ar[dist] ^= ar2[dist];
        dist += stride;
    }
}

template <int REPEAT>
__device__ __inline__ void XOR(uint32_t* ar, uint32_t* ar2, uint32_t stride){
    uint32_t dist = 0;
    for(int i=0; i<REPEAT; i++){
        ar[dist] ^= ar2[dist];
        dist += stride;
    }
}

template <int REPEAT>
__device__ __inline__ void atomic_XOR(uint32_t* dest, uint32_t* src, uint32_t stride){
    uint32_t dist = 0;
    for(int i=0; i<REPEAT; i++){
        atomicXor(dest+2*i,src[dist]);
        atomicXor(dest+2*i+1,src[dist+1]);
        dist+=stride;
    }
}

__device__ inline void XOR(volatile uint64_t* ar, uint32_t stride, const Int128& input){
    ar[0] ^= input.m.x;
    ar[stride] ^= input.m.y;
}

__device__ inline void XOR(volatile uint64_t* ar, uint32_t stride, const Int256& input){
    ar[0] ^= input.m.x;
    ar[stride] ^= input.m.y;
    ar[stride*2] ^= input.m.z;
    ar[stride*3] ^= input.m.w;
}

template <int REPEAT>
__device__ inline void XOR(volatile uint64_t* ar, volatile uint64_t* ar2, uint32_t stride){
    uint32_t dist = 0;
    for(int i=0; i<REPEAT; i++){
        ar[dist] ^= ar2[dist];
        dist += stride;
    }
}

template <int REPEAT>
__device__ inline void atomic_XOR(uint32_t* dest, volatile uint32_t* src, uint32_t stride){
    uint32_t dist = 0;
    for(int i=0; i<REPEAT; i++){
        atomicXor(dest+2*i,src[dist]);
        atomicXor(dest+2*i+1,src[dist+1]);
        dist+=stride;
    }
}

__device__ __inline__ void atomic_xor(uint32_t* ptr, const uint64_t &value,
        uint32_t stride) {
    atomicXor(ptr, static_cast<uint32_t> (value));
    ptr += stride;
    atomicXor(ptr, static_cast<uint32_t> (value >> 32));
}

__device__ __inline__ void atomic_xor(uint32_t* ptr, const Int128& value,
        uint32_t stride) {
    atomicXor(ptr, static_cast<uint32_t> (value.m.x));
    ptr += stride;
    atomicXor(ptr, static_cast<uint32_t> (value.m.x >> 32));
    ptr += stride;
    atomicXor(ptr, static_cast<uint32_t> (value.m.y));
    ptr += stride;
    atomicXor(ptr, static_cast<uint32_t> (value.m.y >> 32));
}

__device__ __inline__ void atomic_xor(uint32_t* ptr, const Int256& value,
        uint32_t stride) {
    atomicXor(ptr, static_cast<uint32_t> (value.m.x));
    ptr += stride;
    atomicXor(ptr, static_cast<uint32_t> (value.m.x >> 32));
    ptr += stride;
    atomicXor(ptr, static_cast<uint32_t> (value.m.y));
    ptr += stride;
    atomicXor(ptr, static_cast<uint32_t> (value.m.y >> 32));
    ptr += stride;
    atomicXor(ptr, static_cast<uint32_t> (value.m.z));
    ptr += stride;
    atomicXor(ptr, static_cast<uint32_t> (value.m.z >> 32));
    ptr += stride;
    atomicXor(ptr, static_cast<uint32_t> (value.m.w));
    ptr += stride;
    atomicXor(ptr, static_cast<uint32_t> (value.m.w >> 32));
}

#endif /* XOR_H_ */
