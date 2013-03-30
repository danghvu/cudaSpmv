/*
 * vector_gen.h
 *
 *  Created on: Dec 17, 2010
 *      Author: hans
 */

#ifndef VECTOR_GEN_H_
#define VECTOR_GEN_H_

#include <cstdlib>
#include <stdint.h>
#include <thrust/host_vector.h>

template <class T>
T myRandom();

template <>
uint32_t myRandom<uint32_t>(){
    return rand();
}

template <>
float myRandom<float>() {
    return (float) rand();
}

template <>
double myRandom<double>() {
     return (double)rand()/(double)RAND_MAX;
}

template <>
uint64_t myRandom<uint64_t>(){
    return (static_cast<uint64_t>(rand()) << 32) | rand();
}

template <>
Int128 myRandom<Int128>(){
    return Int128(myRandom<uint64_t>(), myRandom<uint64_t>());
}

template <>
Int256 myRandom<Int256>(){
    return Int256(make_ulong4(myRandom<uint64_t>(), myRandom<uint64_t>(), myRandom<uint64_t>(), myRandom<uint64_t>()));
}

template <class T>
void generateVector(vector<T> &vec, uint32_t vlen){
	vec.reserve(vlen);
    for(int i=0; i<vlen; i++){
        T random = myRandom<T>();
        vec.push_back(random);
    }
}

template <class T>
void generateVector(thrust::host_vector<T> &vec, uint32_t vlen){
	vec.reserve(vlen);
    for(int i=0; i<vlen; i++){
        T random = myRandom<T>();
        vec.push_back(random);
    }
}


#endif /* VECTOR_GEN_H_ */
