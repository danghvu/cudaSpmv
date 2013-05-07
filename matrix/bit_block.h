/*
 * bit_block.h
 *
 *  Created on: Dec 10, 2010
 *      Author: hans
 */

#ifndef BIT_BLOCK_H_
#define BIT_BLOCK_H_

#include <stdint.h>
#include <iostream>
#include <string>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <vector>
#include <cstdlib>
#include <sstream>
#include "binary_matrix.h"
#include "bit_block_kernel.cuh"
#include "../util/util.h"
#include "../util/textures.h"
#include "../util/data_output_stream.h"
#include "../util/data_input_stream.h"

namespace cuda_bwc{

    template <typename T>
        class BitBlock : public BinaryMatrix<T>{
            public:
                BitBlock() : BinaryMatrix<T>(0,0), num_rows_per_block(WARP_SIZE), num_blocks(0){};
                BitBlock(uint32_t num_rows, uint32_t num_cols) :
                    BinaryMatrix<T>(num_rows,num_cols),
                    num_rows_per_block(WARP_SIZE),
                    num_blocks(divideAndCeil(num_rows, num_rows_per_block)){}
                void set_num_rows(uint32_t nr);
                void readMatrix(MatrixInput &in);
                void writeMatrix(std::ostream &out);
                void readCache(std::istream &in);
                void buildCache(std::ostream &out);
                std::string getCacheName();
                uint64_t memoryUsage();
                void multiply(const T *v, T *r);
                uint32_t granularity();
                uint32_t nonZeroes();

            protected:
                void transferToDevice();

            private:
                thrust::host_vector<uint32_t> hv;
                thrust::host_vector<T> value;

                thrust::device_vector<uint32_t> dv; //sizeof(uint32_t) must be >= WARP_SIZE;
                thrust::device_vector<T> dv_value;

                uint32_t num_rows_per_block;
                uint32_t num_blocks;
                uint32_t non_zeroes;
                static const uint32_t THREADS_PER_BLOCK = 1024;
                //		static const uint32_t MAX_BITS = (SHARED_MEMORY_SIZE / THREADS_PER_BLOCK) / sizeof(T) * WARP_SIZE;
        };

    template <typename T>
        void BitBlock<T>::set_num_rows(uint32_t nr){
            this->num_rows = nr;
            num_blocks = divideAndCeil(this->num_rows, num_rows_per_block);
        }

    template <typename T>
        std::string BitBlock<T>::getCacheName(){
            std::stringstream ss;
            ss << "-bitblock-" << this->num_rows << "-" << this->num_cols << ".bin";
            return ss.str();
        }

    template <typename T>
        void BitBlock<T>::readMatrix(MatrixInput &in){
            uint32_t num_cols = this->num_cols;
            uint32_t num_rows = this->num_rows;
            hv.resize(num_cols*num_blocks, 0);
            value.resize(num_cols*num_blocks*32, 0);    

            uint32_t nz = 0;
            for(uint32_t i=0; i<num_blocks; i++){
                uint32_t limit = min(num_rows_per_block, num_rows-i*num_rows_per_block);
                for(uint32_t j=0; j<limit; j++){
                    uint32_t mask = 1<<j;
                    uint32_t nc;
                    in >> nc;
                    nz += nc;
                    for(uint32_t k=0; k<nc; k++){
                        uint32_t temp;
                        in >> temp;
                        hv[i*num_cols + temp] |= mask;
                        T v = (T) in.getValue();
                        value[32*(i*num_cols + temp) + j] = v;
                    }
                }
            }
            non_zeroes = nz;
            cerr << " Total Read : " << nz << endl;
            transferToDevice();
        }
    template <typename T>
        void BitBlock<T>::writeMatrix(std::ostream &out){
            uint32_t num_cols = this->num_cols;
            uint32_t num_rows = this->num_rows;
            std::vector<uint32_t> vec[num_rows_per_block];
            for(uint32_t i=0; i<num_blocks; i++){
                uint32_t limit = min(num_rows_per_block, num_rows-i*num_rows_per_block);
                for(uint32_t k=0; k<num_cols; k++){
                    uint32_t bits = hv[i*num_cols + k];
                    for(int n=0; n<limit; n++){
                        if(bits & (1<<n)){
                            vec[n].push_back(k);
                        }
                    }
                }
                for(uint32_t j=0; j<limit; j++){
                    out << vec[j].size();
                    for(uint32_t k=0; k<vec[j].size(); k++){
                        out << " " << vec[j][k];
                    }
                    out << std::endl;
                    vec[j].clear();
                }
            }
        }

    template <typename T>
        void BitBlock<T>::readCache(std::istream &in){
            DataInputStream dis(in);
            dis >> this->num_rows >> this->num_cols >> this->non_zeroes;
            dis.readVector(hv);
            dis.readVector(value);
            num_rows_per_block = WARP_SIZE;
            num_blocks = divideAndCeil(this->num_rows, num_rows_per_block);
            transferToDevice();
        }

    template <typename T>
        void BitBlock<T>::transferToDevice(){
            dv.assign(hv.begin(), hv.end());
            dv_value.assign(value.begin(), value.end());
        }

    template <typename T>
        void BitBlock<T>::buildCache(std::ostream &out){
            DataOutputStream dos(out);
            dos << this->num_rows << this->num_cols << this->non_zeroes;
            dos.writeVector(hv);
            dos.writeVector(value);
        }
    template <typename T>
        uint64_t BitBlock<T>::memoryUsage(){
            return hv.size()*sizeof(uint32_t);
        }

    template <typename T>
        uint32_t BitBlock<T>::granularity(){
            return WARP_SIZE;
        }

    template <typename T>
        uint32_t BitBlock<T>::nonZeroes(){
            return non_zeroes;
        }

    template <typename T>
        void BitBlock<T>::multiply(const T *v, T *r){
            static const uint32_t NUM_BLOCKS = getNumMultiprocessors();
            cudaMemset(r, 0, this->num_rows*sizeof(T));
            checkCUDAError("emptying the rows for bitblock format's result");
            bind_x(v);
            bitBlockKernel_32_64<T, THREADS_PER_BLOCK> <<< NUM_BLOCKS, THREADS_PER_BLOCK>>>
                (this->num_rows,
                 this->num_cols,
                 thrust::raw_pointer_cast(&dv[0]),
                 thrust::raw_pointer_cast(&dv_value[0]),
                 v, r);
            unbind_x(v);
        }

    template <>
        void BitBlock<Int128>::multiply(const Int128 *v, Int128 *r){
            static const uint32_t NUM_BLOCKS = getNumMultiprocessors();
            cudaMemset(r, 0, this->num_rows*sizeof(Int128));
            checkCUDAError("emptying the rows for bitblock format's result");
            bitBlockKernel_128_256<Int128, THREADS_PER_BLOCK> <<< NUM_BLOCKS, THREADS_PER_BLOCK>>>
                (this->num_rows,
                 this->num_cols,
                 thrust::raw_pointer_cast(&dv[0]),
                 v, r);
        }

    template <>
        void BitBlock<Int256>::multiply(const Int256 *v, Int256 *r){
            static const uint32_t NUM_BLOCKS = getNumMultiprocessors();
            cudaMemset(r, 0, this->num_rows*sizeof(Int256));
            checkCUDAError("emptying the rows for bitblock format's result");
            bitBlockKernel_128_256<Int256, THREADS_PER_BLOCK> <<< NUM_BLOCKS, THREADS_PER_BLOCK>>>
                (this->num_rows,
                 this->num_cols,
                 thrust::raw_pointer_cast(&dv[0]),
                 v, r);
        }


}

#endif /* BIT_BLOCK_H_ */
