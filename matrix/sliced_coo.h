#ifndef SLICED_COO_H_
#define SLICED_COO_H_

#include "../util/util.h"
#include "../util/data_input_stream.h"
#include "../util/data_output_stream.h"

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <sstream>
#include "matrix.h"
#include "sliced_coo_kernel.h"
#include "../util/textures.h"

template<typename T, const uint32_t LANE_SIZE>
class SlicedCoo: public Matrix<T>{
    public:
        SlicedCoo():Matrix<T>(0,0){};
        SlicedCoo(uint32_t num_rows, uint32_t num_cols):Matrix<T>(num_rows, num_cols){};

        static const uint32_t GROUP_SIZE = (1<<30);

        uint32_t numRowsPerSlice(){return NUM_ROWS_PER_SLICE;};
        uint32_t getLaneSize() { return LANE_SIZE; }

        void readMatrix(MatrixInput &in, vector<uint32_t> &perm);
        void readCache(std::istream &in);
        void buildCache(std::ostream &out);
        std::string getCacheName();

        uint64_t memoryUsage();
        void multiply(const T *v, T *r, cudaStream_t t=0);
        uint32_t granularity();
        uint32_t nonZeroes();

        void transferToDevice();

    private:
        static const uint32_t THREADS_PER_BLOCK = 1024;
        static const uint32_t NUM_ROWS_PER_SLICE = SHARED_MEMORY_SIZE/sizeof(T)/LANE_SIZE;

        uint32_t non_zeroes;
        thrust::host_vector<uint16_t> row;
        thrust::host_vector<uint32_t> col;
        thrust::host_vector<uint32_t> offsets;
        thrust::host_vector<T> value;

        thrust::device_vector<uint32_t> dv_col;
        thrust::device_vector<uint16_t> dv_row; 
        thrust::device_vector<uint32_t> dv_offsets;
        thrust::device_vector<T> dv_value;

};

template<typename T, uint32_t LANE_SIZE>
uint64_t SlicedCoo<T, LANE_SIZE>:: memoryUsage(){
    return col.size()*sizeof(col[0]) + row.size()*sizeof(row[0]) + offsets.size()*sizeof(offsets[0]) + value.size()*sizeof(value[0]);
}

template<typename T, uint32_t LANE_SIZE>
uint32_t SlicedCoo<T, LANE_SIZE>::granularity(){
    return numRowsPerSlice();
}

template<typename T, uint32_t LANE_SIZE>
uint32_t SlicedCoo<T, LANE_SIZE>::nonZeroes(){
    return non_zeroes;
}

template<typename T, uint32_t LANE_SIZE>
void SlicedCoo<T, LANE_SIZE>::readCache(std::istream &in){
    DataInputStream dis(in);
    dis >> (this->num_rows) >> (this->num_cols) >> (non_zeroes);
    dis.readVector(col);
    dis.readVector(row);
    dis.readVector(value);
    dis.readVector(offsets);

    transferToDevice();
}

template<typename T, uint32_t LANE_SIZE>
void SlicedCoo<T, LANE_SIZE>::buildCache(std::ostream &out){
    DataOutputStream dos(out);
    dos << this->num_rows << this->num_cols << non_zeroes;
    dos.writeVector(col);
    dos.writeVector(row);
    dos.writeVector(value);
    dos.writeVector(offsets);
}

template <typename T, uint32_t LANE_SIZE>
void SlicedCoo<T, LANE_SIZE>::transferToDevice(){
    dv_col.assign(col.begin(), col.end());
    dv_row.assign(row.begin(), row.end());
    dv_value.assign(value.begin(), value.end());
    dv_offsets.assign(offsets.begin(), offsets.end());
}


template <typename T, uint32_t LANE_SIZE>
std::string SlicedCoo<T, LANE_SIZE>::getCacheName(){
    std::stringstream ss;
    ss << "-" << LANE_SIZE << "scoo-" << NUM_ROWS_PER_SLICE << "pack-" << this->num_rows << "-" << this->num_cols << ".bin";
    return ss.str();
}

template<typename T, uint32_t LANE_SIZE>
void SlicedCoo<T, LANE_SIZE>::readMatrix(MatrixInput &in, vector<uint32_t> &perm){
    vector < pair<uint32_t, pair< uint16_t, T > > > slice;
    uint32_t nz = 0;
    double dist=0,ndist=0;
    vector<uint32_t> revp(perm.size());
    for (uint32_t i=0;i<perm.size();i++) {
        revp[perm[i]] = i;
    }

    for (uint32_t i = 0; i < this->num_rows;) {
        uint32_t limit = min(this->num_rows, i + numRowsPerSlice());
        slice.clear();
        for (; i < limit; i++) {
            uint32_t temp;
            in >> temp;
            nz += temp;
            for (uint32_t j = 0; j < temp; j++) {
                uint32_t c;
                in >> c;
                if (this->sort) 
                    c = revp[c]; 
                T v;
                v = (T) in.getValue();
                slice.push_back(make_pair(c, make_pair( static_cast<uint16_t>(i % numRowsPerSlice() ), v  )));
            }
        }
        sort(slice.begin(), slice.end());

        offsets.push_back(col.size());
        for (uint32_t j = 0; j < slice.size(); j++) {
            uint32_t c = slice[j].first;
            uint16_t r = slice[j].second.first;
            T v = slice[j].second.second;

            if (col.size() > 0) {
                dist += 1 / (1+std::abs((double) col.back() - c));
                ndist++;
            }

            col.push_back(c);
            row.push_back(r);
            value.push_back(v);
        }
    }
    offsets.push_back(col.size());
    non_zeroes = nz;
    this->setStat( ndist / dist );
}

template<typename T, const uint32_t LANE_SIZE>
void SlicedCoo<T, LANE_SIZE>::multiply(const T *v, T *r, cudaStream_t t){
    //	static const uint32_t NUM_BLOCKS = getNumMultiprocessors();
    bind_x(v);
    // cerr << divideAndCeil(this->num_rows, NUM_ROWS_PER_SLICE) << endl;
    sliced_coo_kernel_32<T, THREADS_PER_BLOCK, NUM_ROWS_PER_SLICE, LANE_SIZE> <<<divideAndCeil(this->num_rows, NUM_ROWS_PER_SLICE), THREADS_PER_BLOCK, 0, t>>>( //dim3(THREADS_PER_BLOCK/4, 4)>>>(
        this->num_rows,
        divideAndCeil(this->num_rows, NUM_ROWS_PER_SLICE),
        thrust::raw_pointer_cast(&this->dv_col[0]),
        thrust::raw_pointer_cast(&this->dv_row[0]),
        thrust::raw_pointer_cast(&this->dv_value[0]),
        thrust::raw_pointer_cast(&this->dv_offsets[0]),
        v,
        r);
            unbind_x(v);
            }

#endif /* SLICED_COO_H_ */
