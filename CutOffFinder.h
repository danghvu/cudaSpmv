#ifndef CUTOFFFINDER_H
#define CUTOFFFINDER_H

#include "matrix/MatrixInput.h"
#include "BalancedMatrix.h" 
#include "matrix/factory.h" 

#include <thrust/device_vector.h>

#include <cuda_runtime_api.h>
#include <sys/time.h>

#include <iostream>
#include <numeric>
#include <stdexcept>
#include <vector>
#include <stdint.h>
#include <numeric>
#include <queue>
#include "util/timer.h"


#define gnzps(a,b,c) ( (a) / (b) / 1e9*(c) )

using namespace std;

struct Result {
    string format;
    uint32_t startingRow;
    uint32_t numRows;
    uint32_t rowsPerSlice;
    double avgDist;

    double time;
    uint64_t nnz;
    double gnzps;
    uint32_t ntimes;

    string toString() {
        ostringstream ostr;
        ostr << format << " - row " << startingRow << ", " << numRows
                << " rows, " << rowsPerSlice << " rows per slice, " << gnzps
                << " gnz/s, " << time << " s, " << nnz << " nnz, avg d:" << avgDist;
        return ostr.str();
    }
};

const uint32_t FORMATS_LEN = 9;
const uint32_t LOOK_AHEAD = 9;
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

        thrust::device_vector<ValueType> g_input, g_output;
    
    public:
        
        CutOffFinder( MMFormatInput &f ): mf(f), mat(f.data) {
        
            prefsum.resize( mat.num_rows, 0 );
            prefsum[ mf.perm[0] ] = mat.row_offsets[ mf.perm[0] + 1 ] - mat.row_offsets[ mf.perm[0] ];

            for (int i=1;i<mat.num_rows;i++)  {
                prefsum[ mf.perm[i] ] = prefsum[ mf.perm[i-1] ] + getCSRNumRows(mat, mf.perm[i] ); //(mat.row_offsets[ mf.perm[i] + 1] - mat.row_offsets[ mf.perm[i] ]); 
            }
        }
        
        Result getPerformance(const string &fmt, uint32_t currentRow);

        void execute_trial(bool verbose);
        Result getPerformance_trial(const string &fmt, uint32_t currentRow);

        void execute(bool verbose);
        void balance( const CutOffFinderOutput &cutOffOutput,  uint32_t numProcs,  vector<uint32_t> &permutation);

        void printResult(char *filename);
        void writeCache(char *filename);
};

template<class T>
Result CutOffFinder<T>::getPerformance_trial(const string &fmt, uint32_t currentRow) {
    uint32_t numProcs = getNumMultiprocessors();
    Matrix<T>* bm = matrix_factory::getMatrixObject<T>(fmt);
    bm->set_num_cols(mat.num_cols);

    Result result;
    if (fmt.rfind("scoo") != string::npos) {
        uint32_t minRows = bm->granularity() * numProcs;
        if (mat.num_rows - currentRow < minRows) {
            minRows = mat.num_rows - currentRow;
        }
        bm->set_num_rows(minRows);
        BalancedMatrix balM(mat, mf.perm, currentRow, minRows, numProcs, bm->granularity());
        VectorOfVectorMatrixInput vmi(balM.matrix, balM.permutation);
        bm->readMatrix(vmi);
    } else { // this is not for scoo
        uint32_t minRows = bm->granularity() * numProcs * 32;
        if (mat.num_rows - currentRow < minRows)
            minRows = mat.num_rows - currentRow;
        bm->set_num_rows(minRows);
        VectorOfVectorMatrixInput vmi(mat, mf.perm, bm->get_num_rows(), currentRow);
        bm->readMatrix(vmi);
    }

    bm->transferToDevice();

    timeval now, then;
    gettimeofday(&then, NULL);
    for (int k = 0; k < NTIMES; k++) {
        bm->multiply(thrust::raw_pointer_cast(&g_input[0]), thrust::raw_pointer_cast(&g_output[0]));
    }
    cudaDeviceSynchronize();
    gettimeofday(&now, NULL);

    result.avgDist = bm->getStat();
    result.format = fmt;
    result.nnz = bm->nonZeroes();
    result.ntimes = NTIMES;
    result.time = (now.tv_sec - then.tv_sec + 1e-6 * (now.tv_usec
            - then.tv_usec));
    result.gnzps = gnzps( result.nnz , result.time , NTIMES) ;
    result.startingRow = currentRow;
    result.numRows = bm->get_num_rows();
    result.rowsPerSlice = bm->granularity();

    delete bm;

    return result;
}


template<typename T>
void CutOffFinder<T>::execute_trial( bool verbose) {
    cerr << " --- on execute --- " << endl;

    this->numProcs = getNumMultiprocessors();
    g_input.resize(max(mat.num_rows, mat.num_cols),0.5);
    g_output.resize(max(mat.num_rows, mat.num_cols),0);

    uint32_t currentRow = 0;
    vector<Result> resultVector;

    for (uint32_t i = 0; i < FORMATS_LEN - 1;) {
        cout << "Getting result for " << FORMATS[i] << endl;
        Result result = getPerformance_trial(FORMATS[i], currentRow);
        int buf = 0; 
        cout << "Comparing " << result.toString() << endl;
        cout << "With:" << endl;
        for (uint32_t j = i + 1; j < FORMATS_LEN; j++) {
            Result comp = getPerformance_trial(FORMATS[j], currentRow);
            cout << comp.toString() << endl;
            if (result.gnzps >= comp.gnzps && buf < 2) {
                break;
            } else {
                if (result.gnzps >= comp.gnzps) { buf++; continue; } 
                if (j >= LOOK_AHEAD) { //This if statement is to check whether xlscoo is better if used later
                    Result nextComp = getPerformance_trial( FORMATS[j],
                            currentRow + result.numRows);
                    cout << "Look ahead " << nextComp.toString() << endl;
                    if ( gnzps( result.nnz + nextComp.nnz , result.time + nextComp.time , NTIMES )  > comp.gnzps) {
                        //this means it's better to use xlscoo at a later point
                        break;
                    }
                }
                result = comp;
                i = j;
            }
        }
        cout << "Chosen: " << result.toString() << endl;
        resultVector.push_back(result);
        currentRow += result.numRows;
        if (currentRow >= mat.num_rows) break;
    }

    cout << "Result:" << endl;
    for (uint32_t i = 0; i < resultVector.size(); i++) {
        cout << resultVector[i].toString() << endl;
        output.formats.push_back(resultVector[i].format);
        output.startRows.push_back(resultVector[i].startingRow);
        output.numRows.push_back(resultVector[i].numRows);
        output.rowsPerSlice.push_back(resultVector[i].rowsPerSlice);
    }
    cout << endl;

    string lastformat = output.formats.back();

    uint32_t curNumRows = accumulate(output.numRows.begin(), output.numRows.end(), 0);
    Matrix<T>* xlscoo = matrix_factory::getMatrixObject<T>( lastformat );
    for(int i=0; i<2; i++){
        uint32_t nr = xlscoo->granularity()*this->numProcs;
        if(nr + curNumRows <= mat.num_rows){
            output.formats.push_back(lastformat);
            output.startRows.push_back(curNumRows);
            output.numRows.push_back(nr);
            output.rowsPerSlice.push_back(xlscoo->granularity());
            cout << "Adding " << lastformat << " from row " << curNumRows << " with " << nr << " rows" << endl;
            curNumRows += nr;
        } else{
            break;
        }
    }
    delete xlscoo;

    balance(output, this->numProcs, new_perm); 
    
    cerr << " --- done execute --- " << endl;
}


template<class T>
Result CutOffFinder<T>::getPerformance(const string &fmt, uint32_t currentRow) {

    Result result;
    if (fmt.rfind("scoo")==string::npos) {
       //sle
        Matrix<T>* bm = matrix_factory::getMatrixObject<T>(fmt);
        uint32_t minRows = bm->granularity();
        if (mat.num_rows - currentRow < minRows) 
            minRows = mat.num_rows - currentRow;

        double diff = 0;
        int prevl;
        int nnz=0;
        for (int i=0; i<minRows;i++){
            int l = getCSRNumRows( mat , mf.perm[currentRow + i] );
            nnz += l;
            if (i>0) diff += (prevl - l);
            prevl = l;
        } 
        diff /= (minRows - 1);
        if (diff <= 8)
            result.avgDist = 1;
        else 
            result.avgDist = 0;
        result.format =fmt;
        result.nnz = nnz;
        result.startingRow = currentRow;
        result.numRows = minRows;
        result.rowsPerSlice = minRows;    
        return result;
    }

    Matrix<T>* bm = matrix_factory::getMatrixObject<T>(fmt);

    int numProcs = this->numProcs;

    uint32_t minRows = bm->granularity() * numProcs;


    if (mat.num_rows - currentRow < minRows) {
        minRows = mat.num_rows - currentRow;
    }

    uint32_t nnz = 0;
//    for (int i=0;i<minRows; i++) 
//       nnz += mat[perm[currentRow+i]].size();
    nnz = prefsum[ mf.perm[currentRow + minRows - 1] ];
    if (currentRow > 0) 
        nnz -= prefsum[ mf.perm[currentRow - 1] ];

    result.avgDist = ( nnz / ((double) SHARED_MEMORY_SIZE / sizeof(T)) / numProcs );
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
    cerr << " executing ... ";
    uint32_t currentRow = 0;
    vector<Result> resultVector;
    
    this->numProcs = getNumMultiprocessors();

    double min_rows_per_shared = (double) mat.num_entries / (double) mat.num_rows;
    double max_rows_per_shared = 2 * min_rows_per_shared;

    if (verbose) cerr << "MIN_ROWS_PER_SHARED: " << min_rows_per_shared << endl;
    if (verbose) cerr << "MAX_ROWS_PER_SHARED: " << max_rows_per_shared << endl;

    for (uint32_t i = 0; i < FORMATS_LEN - 1;) {

        Result result = getPerformance("sle", currentRow);

        if (!result.avgDist) {

        if (verbose) cerr << "Getting result for " << FORMATS[i] << endl;        

        result = getPerformance(FORMATS[i], currentRow);
     
        if (verbose) cerr << "Evaluate " << result.toString() << endl;
        if (result.avgDist < max_rows_per_shared)
        for (uint32_t j = i + 1; j < FORMATS_LEN; j++) {
            Result comp = getPerformance(FORMATS[j], currentRow);
            if (verbose) cerr << comp.toString() << endl;
            if (comp.avgDist > max_rows_per_shared && result.avgDist > min_rows_per_shared) break;
            if (comp.avgDist > result.avgDist) {
                result = comp;
                i=j;
            }
        }

        }
        if (verbose) cerr << "Chosen: " << result.toString() << endl;
        resultVector.push_back(result);
        currentRow += result.numRows;
        if (currentRow >= mat.num_rows) break;
    }

    if (verbose)    cerr << "Result:" << endl;
    for (uint32_t i = 0; i < resultVector.size(); i++) {
        if (verbose) cerr << resultVector[i].toString() << endl;
        output.formats.push_back(resultVector[i].format);
        output.startRows.push_back(resultVector[i].startingRow);
        output.numRows.push_back(resultVector[i].numRows);
        output.rowsPerSlice.push_back(resultVector[i].rowsPerSlice);
    }
    if (verbose) cerr << endl;
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
            if (verbose) cerr << "Adding " << lastformat << " from row " << curNumRows << " with " << nr << " rows" << endl;
            curNumRows += nr;
        } else{
            break;
        }
    }
    delete lastf;

    balance(output, numProcs ,new_perm); 
    
    cerr << " OK " << endl;
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
//      cout << "Base row: " << baseRow << endl;
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
void CutOffFinder<T>::printResult(char * mfile) { 
    cerr << " printResult ... " ;
//    uint32_t maxRowsPerFormat = 10e8;

    char outfile[255];

    sprintf(outfile, FORMAT_PERM, mfile);
    ofstream permout( outfile );
    DataOutputStream out2(permout);
    out2.writeVector( new_perm );
    permout.close();

    sprintf(outfile, FORMAT_OUTPUT, mfile);
    
    ofstream out;
    out.open(outfile);
    out.exceptions(ios::failbit | ios::badbit);

//    out << new_perm.size() ;
//    for (int i=0; i<new_perm.size();i++)
//        out << " " << new_perm[i];
//    out << endl;

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

    cerr << " OK " << endl;
}

template <class T>
void CutOffFinder<T>::writeCache(char* output_path){
    cerr << " writing cache ... ";
    uint32_t currentRow = 0;

    output.startRows.push_back(mat.num_rows);
    mf.perm = new_perm;  

    for(int i=0; i<output.startRows.size()-1; i++){
        Matrix<T> *bm = matrix_factory::getMatrixObject<T>(output.formats[i]);
        bm->set_num_cols(mat.num_cols);
        bm->set_num_rows(output.startRows[i+1] - output.startRows[i]);

        cerr << "Building cache for " << output.formats[i] << endl;
        bm->readMatrix(mf);
        stringstream cacheOutput;
        cacheOutput << output_path << "-" << currentRow << bm->getCacheName();
        cerr << "Writing cache to " << cacheOutput.str() << endl;
        ofstream out(cacheOutput.str().c_str());
        out.exceptions(ios_base::failbit | ios_base::badbit);
        bm->buildCache(out);
        out.close();

        currentRow += bm->get_num_rows();
        delete bm;
    }
    cerr << " OK " << endl;
}
#endif 
