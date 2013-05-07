/* vim: set ft=cpp: */

#define CUSP_USE_TEXTURE_MEMORY

#include <stdlib.h>
#include <cusp/hyb_matrix.h>
#include <cusp/csr_matrix.h>
#include <cusp/coo_matrix.h>
#include <cusp/dia_matrix.h>
#include <cusp/ell_matrix.h>
#include <cusp/multiply.h>
#include <cusp/io/matrix_market.h>
#include <cusp/print.h>
#include <iostream>
#include <fstream>

#define DEFAULT_VECTOR_VALUE 1

using namespace std;

cusp::csr_matrix<int, float, cusp::host_memory> B;
cusp::array1d<float, cusp::host_memory> v;
string matrix_path, vector_path;
int n=10;
int block=32;
int gpu = 0;
bool print =false;

#define bench(format) \
    cusp::array1d<T, cusp::device_memory> x[2];\
    try { \
FREECHECK \
    A = B ;\
ENDFREECHECK\
    x[0] = v; \
    x[1].resize(A.num_rows, 0);\
    } catch (cusp::exception) { \
        cerr << "! failed: unable to convert csr to " << format << endl;\
        return;\
    }\
    cudaEvent_t start, stop;\
    float time;\
    cudaEventCreate(&start); \
    cudaEventCreate(&stop); \
    cudaEventRecord( start, 0 ); \
    for (int i=0; i<n; i++) { \
        cusp::blas::fill(x[1-i&1],0);\
        cusp::multiply(A, x[i&1],x[1-i&1]); \
    } \
    cudaDeviceSynchronize(); \
    cudaEventRecord( stop, 0 ); \
    cudaEventSynchronize( stop ); \
    cudaEventElapsedTime( &time, start, stop ); \
    std::cerr << format << ": " << time/1000 << "(s) ---> " << A.num_entries * 2 * n / 10e5 / time << " GFL/s"<< std::endl; 

#define INMB(x) (x)/1024/1024
     
#define FREECHECK \
    size_t free, dvTotal;\
	cudaMemGetInfo	(&free,&dvTotal);

#define ENDFREECHECK \
    cudaDeviceSynchronize();\
    size_t free2;\
    cudaMemGetInfo (&free2, &dvTotal);\
    cerr << " SIZE: " << INMB (free - free2) << endl;  


template<typename T>
void bench_csr() {
    cusp::csr_matrix<int, T, cusp::device_memory> A;
    bench("CSR");
    if (print) {
        v = x[n&1];
        for (int i=0;i<B.num_rows;i++) 
            cout << v[i] << endl;
    }
}

template<typename T>
void bench_hyb() {
    cusp::hyb_matrix<int, T, cusp::device_memory> A;
    bench("HYB");
}

template<typename T>
void bench_coo() {
    cusp::coo_matrix<int, T, cusp::device_memory> A;
    bench("COO");
}

void print_usage(char **args) {
    cout << "Usage: " << args[0] << " -m <matrix> -n <niteration> -b <blocksize> -g <gpuid> [-s > [outputfile]"
         << endl;
}

template <typename T>
void readVector(const char *file, cusp::array1d<T, cusp::host_memory> &v, uint32_t len){
    ifstream in;
    in.exceptions ( ifstream::failbit | ifstream::badbit );
    try {
        in.open(file);
        for (int i=0;i<len;i++) {
            T t; in >> t;
            v.push_back(t);
        }
    }catch(ifstream::failure e) {
        cout << "Exception opening/reading: " << file << endl;
        throw e;
    }
    in.close();
}

int main(int argc, char **args){
    int opt;
    if (argc < 2) {
       print_usage(args);
       return 0;
    }

    while ((opt = getopt(argc, args, "b:m:v:sn:g:"))!= -1){
        switch (opt){
            case 'b':
                block = atoi(optarg);
                break;
            case 'm': // matrix file
                matrix_path = optarg;
                break;
            case 's':
                print = true;
                break;
            case 'n': // #iterations
                n = atoi(optarg);
                break;
            case 'v': // vector file
                vector_path = optarg;
                break;
            case 'g':
                gpu = atoi(optarg);    
                break;
            default :
                cerr << "unknown argument: " << (char)opt << endl;
                break;
        }
    }

    cerr << "Using GPU id: " << gpu << endl;
    cudaSetDevice(gpu);

    // load a matrix stored in MatrixMarket format
    std::ifstream inf(matrix_path.c_str());
    cusp::io::read_matrix_market_stream(B, inf);

    cerr << B.num_rows << " " << B.num_cols << endl;
    if (vector_path.empty())
        v.resize(max( B.num_cols, B.num_rows ), DEFAULT_VECTOR_VALUE );
    else {
        v.reserve(max( B.num_cols, B.num_rows ));
        readVector(vector_path.c_str(), v, B.num_cols);
    }

    cerr << "Starting ... " << endl;

    if (block == 64) {
        bench_csr<double>();
        bench_coo<double>();
        bench_hyb<double>();
    }
    else {
        bench_csr<float>();
        bench_coo<float>();
        bench_hyb<float>();
    }

    return 0;
}
