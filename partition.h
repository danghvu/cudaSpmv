#ifndef CUTOFFFINDER_H
#define CUTOFFFINDER_H

#include <iostream>
#include <numeric>
#include <stdexcept>
#include <vector>
#include <stdint.h>
#include <numeric>
#include <queue>
#include <thrust/device_vector.h>
#include <cuda_runtime_api.h>
#include <sys/time.h>


#include "util/timer.h"

#include "matrix/matrix_input.h"
#include "matrix/factory.h" 

using namespace std;

struct Slice{
    uint32_t sum;
    uint32_t num_rows;
    std::vector<uint32_t> row_start;

    Slice() : sum(0), num_rows(0){
        row_start.clear();
    }

    void addRow(uint32_t index, uint32_t weight){
        sum += weight;
        num_rows++;
        row_start.push_back(index);
    }

    bool operator()(const Slice* lhs, const Slice* rhs){
        return rhs->sum < lhs->sum;
    }
};


struct Result {
    string format;
    uint32_t startingRow;
    uint32_t numRows;
    uint32_t rowsPerSlice;
    double avgDist;

    double time;
    uint64_t nnz;
    uint32_t ntimes;

    string toString() {
        ostringstream ostr;
        ostr << format << " - row " << startingRow << ", " << numRows
             << " rows, " << rowsPerSlice << " rows per slice, " << nnz << " nnz, avg d:" << avgDist;
        return ostr.str();
    }
};

const uint32_t NTIMES = 10;
const string FORMATS[] = { // matrix_factory::BIT_BLOCK,  
    matrix_factory::SLICED_COO_256,	
    matrix_factory::SLICED_COO_128,
    matrix_factory::SLICED_COO_64,
    matrix_factory::SLICED_COO_32,
    matrix_factory::SLICED_COO_16,
    matrix_factory::SLICED_COO_8,
    matrix_factory::SLICED_COO_4,
    matrix_factory::SLICED_COO_2,
    matrix_factory::SLICED_COO_1,
};

uint32_t FORMATS_LEN = (sizeof(FORMATS) / sizeof(FORMATS[0]));


struct CutOffFinderOutput {
    vector<string> formats;
    vector<uint32_t> startRows;
    vector<uint32_t> numRows;
    vector<uint32_t> rowsPerSlice;
};


template <class ValueType>
class CutOffFinder { 
    private:
        csr_matrix &mat;
        uint32_t numProcs;
        MMFormatInput &mf;
        vector<uint32_t> new_perm;

        vector<uint32_t> prefsum;

        CutOffFinderOutput output;

        vector<Matrix<ValueType> *> mfh;

        thrust::device_vector<ValueType> g_input, g_output;

        bool sortcol;
        void test();

    public:

        CutOffFinder( MMFormatInput &f, bool sortcol ): mf(f), mat(f.data) {

            this->sortcol = sortcol;

            prefsum.resize( mat.num_rows, 0 );
            prefsum[ mf.perm[0] ] = mat.row_offsets[ mf.perm[0] + 1 ] - mat.row_offsets[ mf.perm[0] ];

            for (int i=1;i<mat.num_rows;i++)  {
                prefsum[ mf.perm[i] ] = prefsum[ mf.perm[i-1] ] + getCSRNumRows(mat, mf.perm[i] ); //(mat.row_offsets[ mf.perm[i] + 1] - mat.row_offsets[ mf.perm[i] ]); 
            }
        }

        Result getPerformance(const string &fmt, uint32_t currentRow);

        void execute(bool verbose);
        void balance( const CutOffFinderOutput &cutOffOutput,  uint32_t numProcs,  vector<uint32_t> &permutation);

        void printResult(string filename);
        void writeCache(string filename);

        double theta;
};

template<class T>
Result CutOffFinder<T>::getPerformance(const string &fmt, uint32_t currentRow) {

    Result result;
    Matrix<T>* bm = matrix_factory::getMatrixObject<T>(fmt);
    int numProcs = this->numProcs;
    uint32_t minRows = bm->granularity() * numProcs;

    if (mat.num_rows - currentRow < minRows) {
        minRows = mat.num_rows - currentRow;
    }

    uint32_t nnz = 0;
    nnz = prefsum[ mf.perm[currentRow + minRows - 1] ];
    if (currentRow > 0) 
        nnz -= prefsum[ mf.perm[currentRow - 1] ];

    result.avgDist =  (nnz*1.0/minRows) / ( SHARED_MEMORY_SIZE*1.0 / bm->granularity() / sizeof(T) ); // ( nnz / ((double) SHARED_MEMORY_SIZE / sizeof(T)) / numProcs );
    result.format = fmt;
    result.nnz = nnz; 
    result.ntimes = NTIMES;
    result.startingRow = currentRow;
    result.numRows = minRows; 
    result.rowsPerSlice = bm->granularity();

    delete bm;
    return result;
}

template <class T>
void CutOffFinder<T>::execute (bool verbose) {
    cout << " executing ... ";
    uint32_t currentRow = 0;
    vector<Result> resultVector;

    this->numProcs = getNumMultiprocessors();

    double min_per_shared = theta;
    double max_per_shared = theta*2;

    if (verbose) cout << "MIN_ROWS_PER_SHARED: " << min_per_shared << endl;
    if (verbose) cout << "MAX_ROWS_PER_SHARED: " << max_per_shared << endl;

    for (uint32_t i = 0; i < FORMATS_LEN - 1;) {

        if (verbose) cout << "Getting result for " << FORMATS[i] << endl;        

        Result result = getPerformance(FORMATS[i], currentRow);

        if (verbose) cout << "Evaluate " << result.toString() << endl;

        if (result.avgDist < min_per_shared)       
            for (uint32_t j = i + 1; j < FORMATS_LEN; j++) {
                Result comp = getPerformance(FORMATS[j], currentRow);
                if (verbose) cout << comp.toString() << endl;
                if (result.avgDist < min_per_shared){
                    result = comp;
                    i = j;
                    continue;
                }

                if (comp.avgDist >= min_per_shared && comp.avgDist <= max_per_shared) {
                    result = comp;
                    i=j;
                }
                else if (comp.avgDist > max_per_shared) {
                    break;
                }
            }
        if (verbose) cout << "Chosen: " << result.toString() << endl;
        resultVector.push_back(result);
        currentRow += result.numRows;
        if (currentRow >= mat.num_rows) break;
    }

    if (verbose)    cout << "Result:" << endl;
    for (uint32_t i = 0; i < resultVector.size(); i++) {
        if (verbose) cout << resultVector[i].toString() << endl;
        output.formats.push_back(resultVector[i].format);
        output.startRows.push_back(resultVector[i].startingRow);
        output.numRows.push_back(resultVector[i].numRows);
        output.rowsPerSlice.push_back(resultVector[i].rowsPerSlice);
    }
    if (verbose) cout << endl;
    string lastformat = output.formats.back();

    uint32_t curNumRows = accumulate(output.numRows.begin(), output.numRows.end(), 0);
    Matrix<T>* lastf = matrix_factory::getMatrixObject<T>( lastformat );
    for(int i=0; i<2; i++){
        uint32_t nr = lastf->granularity() * numProcs; 
        if(nr + curNumRows <= mat.num_rows){
            output.formats.push_back(lastformat);
            output.startRows.push_back(curNumRows);
            output.numRows.push_back(nr);
            output.rowsPerSlice.push_back(lastf->granularity());
            if (verbose) cout << "Adding " << lastformat << " from row " << curNumRows << " with " << nr << " rows" << endl;
            curNumRows += nr;
        } else{
            break;
        }
    }
    delete lastf;

    balance(output, numProcs ,new_perm); 

    cout << " OK " << endl;
}

template <class T>
void CutOffFinder<T>::balance(
        const CutOffFinderOutput &cutOffOutput,
        uint32_t numProcs,
        vector<uint32_t> &permutation){
    uint32_t baseRow = 0;
    permutation.clear();
    permutation.reserve(max(mat.num_rows, mat.num_cols));
    const vector<uint32_t> &oldPerm = mf.perm; 

    // for each horizontal split 
    uint32_t curOffset = 0;
    // cout << "Base row: " << baseRow << endl;
    for(uint32_t i=0; i<cutOffOutput.formats.size(); i++){
        if(cutOffOutput.formats[i].rfind("scoo") != string::npos){
            int numProcss = numProcs;
            vector<Slice> slices(numProcss);
            priority_queue<Slice*, vector<Slice*>, Slice> pq;
            for(uint32_t j=0; j<numProcss; ++j){
                pq.push(&slices[j]);
            }

            for(uint32_t j=0; j<cutOffOutput.numRows[i]; ++j){
                Slice *sc = pq.top();
                pq.pop();
                uint32_t index = baseRow+curOffset+j;
                sc->addRow(oldPerm[index], mat.row_offsets[oldPerm[index] + 1] - mat.row_offsets[oldPerm[index]] );
                if(sc->num_rows < cutOffOutput.rowsPerSlice[i]){
                    pq.push(sc);
                }
            }

            for(uint32_t j=0; j<numProcss; ++j){
                permutation.insert(permutation.end(), slices[j].row_start.begin(), slices[j].row_start.end());
                //cout << slices[j].sum << " ";
            }
            //cout << endl;

        }else{
            permutation.insert(
                    permutation.end(),
                    oldPerm.begin()+baseRow+curOffset,
                    oldPerm.begin()+baseRow+curOffset+cutOffOutput.numRows[i]);
        }
        curOffset += cutOffOutput.numRows[i];

    }

    uint32_t nextBaseRow = baseRow + max(mat.num_rows, mat.num_cols); 

    permutation.insert(permutation.end(),
            oldPerm.begin()+baseRow+curOffset,
            oldPerm.begin()+nextBaseRow);
    baseRow = nextBaseRow;
}

template <class T>
void CutOffFinder<T>::printResult(string mfile) { 
    cout << " printResult ... " ;
    char outfile[255];

    sprintf(outfile, FORMAT_PERM, mfile.c_str());
    ofstream permout( outfile );
    DataOutputStream out2(permout);
    out2.writeVector( new_perm );
    permout.close();

    sprintf(outfile, FORMAT_OUTPUT, mfile.c_str());

    ofstream out;
    out.open(outfile);
    out.exceptions(ios::failbit | ios::badbit);

    for (int i=1; i<output.formats.size(); ) {
        if (output.formats[i] == output.formats[i-1]){
            output.formats.erase(output.formats.begin() + i);
            output.startRows.erase(output.startRows.begin() + i);
        }
        else i++;
    } 

    out << output.formats.size() << endl;
    copy(output.formats.begin(), output.formats.end(), //+firstLscooLoc,
            ostream_iterator<string> (out, " "));
    out << endl;

    // --- output start row --- //
    copy(output.startRows.begin(), output.startRows.end(), //+ firstLscooLoc,
            ostream_iterator<uint32_t> (out, " "));
    out << endl;

    out.close();

    cout << " OK " << endl;
}

template<class T>
void CutOffFinder<T>::test(){
    g_input.resize(max(mat.num_rows, mat.num_cols),1);
    g_output.resize(max(mat.num_rows, mat.num_cols),0);

    T* v = p(g_input);
    T* r = p(g_output);

    uint32_t cur_row = 0;
    for(uint32_t j=0; j<mfh.size(); j++){
        mfh[j]->multiply(v, r+cur_row);
        cur_row += mfh[j]->get_num_rows();
    }
    cudaDeviceSynchronize();
    checkCUDAError("kernel finish");
}

template <class T>
void CutOffFinder<T>::writeCache(string output_path){
    cout << " writing cache ... ";
    uint32_t currentRow = 0;

    output.startRows.push_back(mat.num_rows);
    mf.perm = new_perm;  

    for(int i=0; i<output.startRows.size()-1; i++){
        Matrix<T> *bm = matrix_factory::getMatrixObject<T>(output.formats[i]);
        bm->set_num_cols(mat.num_cols);
        bm->set_num_rows(output.startRows[i+1] - output.startRows[i]);
        bm->set_sort_col(sortcol);

        cout << "Building cache for " << output.formats[i] << endl;
        bm->readMatrix(mf, new_perm);
        stringstream cacheOutput;
        cacheOutput << output_path << "-" << currentRow << bm->getCacheName();
        cout << "Writing cache to " << cacheOutput.str() << endl;
        ofstream out(cacheOutput.str().c_str());
        out.exceptions(ios_base::failbit | ios_base::badbit);
        bm->buildCache(out);
        out.close();
        bm->transferToDevice();
        currentRow += bm->get_num_rows();
        mfh.push_back(bm);
    }

    cout << "Testing 1 iteration ... " ;
    test();
    cout << " All OK." << endl;
}
#endif 
