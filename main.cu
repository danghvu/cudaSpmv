#include <fstream>
#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include <ctime>
#include <iomanip>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include "matrix/MatrixInput.h"
#include "util/data_input_stream.h"
#include "util/data_output_stream.h"
#include "util/timer.h"
#include "matrix/factory.h"
#include "util/vector_gen.h"
#include <sys/stat.h>
#include "parameters.h"

#define DEFAULT_VECTOR_VALUE 1

using namespace std;
Parameter app;

uint32_t block_size = 32;
uint32_t times = 1;
uint64_t seed = 1;
int gpu = 0;
string matrix_path;
string vector_path;
vector<string> formats;
vector<uint32_t> start_rows;
vector<uint32_t> perm, revperm;
bool rebuild = false;
bool print = false;
bool writeMatrixParts = false;
bool DEBUG = false;

template <typename T>
void readVector(const char *file, thrust::host_vector<T> &v, uint32_t len){
    ifstream in;
    in.exceptions ( ifstream::failbit | ifstream::badbit );
    try {
        in.open(file);
        in >> setbase(16);
        for (int i=0;i<len;i++) {
            T t; in >> t;
            v.push_back(t);
        }
    }catch(ifstream::failure e) {
        cerr << "Exception opening/reading: " << file << endl;
        throw e;
    }
    in.close();
}

template <typename T>
struct MatrixFormatsHolder{
	vector<Matrix<T> *> vec;
	~MatrixFormatsHolder(){
		for(uint32_t i=0; i<vec.size(); i++){
			if(vec[i] != NULL){
				delete vec[i];
			}
		}
	}
};

void skipLines(MMFormatInput &in, uint32_t lines){
	for(uint32_t i=0; i<lines; i++){
		uint32_t nc, temp;
		in >> nc;
		for(uint32_t j=0; j<nc; j++){
			in >> temp;
		}
	}
}

template <typename T>
void execute(){
	uint32_t num_rows, num_cols, cur=0, nnz;
    ifstream f(matrix_path.c_str());
	MMFormatInput matrix_input(f, perm);

	num_rows = matrix_input.getNumRows();
    num_cols = matrix_input.getNumCols();
    nnz = matrix_input.getNNz();

    cerr << num_rows << " " << num_cols << " " << nnz << " " << static_cast<double>(nnz)/num_rows << endl;
	
	//read vector
	thrust::host_vector<T> hv;
	if(vector_path != ""){
		hv.reserve(max(num_rows,num_cols));
		readVector(vector_path.c_str(), hv, num_cols);
	}else{
		//generateVector(hv, max(num_rows,num_cols));
        hv.resize(max(num_rows,num_cols), DEFAULT_VECTOR_VALUE);
	}
	
	if (DEBUG) cerr << "Vector storage: " << hv.size() * sizeof(T) / 1024.0 / 1024.0 << " MB" << endl;
	
	MatrixFormatsHolder<T> mfh; 
	for(uint32_t i=0; i<formats.size(); i++){
		Matrix<T> *matrix = matrix_factory::getMatrixObject<T>(formats[i]);
		if(i>0){
			uint32_t nr = start_rows[i]-start_rows[i-1];
			cur += nr;
			mfh.vec.back()->set_num_rows(nr);
		}
		matrix->set_num_cols(num_cols);
		mfh.vec.push_back(matrix);			
	}

	if (cur >= num_cols) 
        throw ios::failure("Sum of rows > number of rows!");

	mfh.vec.back()->set_num_rows(num_rows - cur);
	uint32_t skip = 0;
	for(uint32_t i=0; i<mfh.vec.size(); i++){
		struct stat st_file_info;
		stringstream ss;
		ss << matrix_path << "-" << start_rows[i] << mfh.vec[i]->getCacheName();
		string s(ss.str());
		if(!rebuild && !stat(s.c_str(), &st_file_info)){
			ifstream inCache(s.c_str());
			if (DEBUG) cerr<<"Reading cache from row " << start_rows[i] << endl;
			mfh.vec[i]->readCache(inCache);
			inCache.close();
			skip += mfh.vec[i]->get_num_rows();
		}else{
			skipLines(matrix_input, skip);
			skip = 0;
			if (DEBUG) cerr << "Reading matrix from row " << start_rows[i] << endl;
			mfh.vec[i]->readMatrix(matrix_input);
			ofstream outCache(s.c_str());
			if (DEBUG) cerr << "Building cache... " << endl;
			mfh.vec[i]->buildCache(outCache);
			outCache.close();
		}
        mfh.vec[i]->transferToDevice();
	}
	f.close();

	double total = 0;
	for(uint32_t j=0; j<mfh.vec.size(); j++){
		double cur = mfh.vec[j]->memoryUsage()/1024.0/1024.0;
		total += cur;
	}
	
	thrust::device_vector<T> dv[2];
	dv[0].assign(hv.begin(), hv.end());
	dv[1].resize(hv.size(), 0);
	
	size_t free, dvTotal;
	cudaMemGetInfo	(&free,&dvTotal);
    if (DEBUG){
        cerr << "Used device memory: " << (dvTotal-free)/1024.0/1024.0 << "MB" <<  endl;
        cerr << "Free device memory: " << free/1024.0/1024.0 << "MB" << endl;
        cerr << "Total device memory: " << dvTotal/1024.0/1024.0 << "MB" <<  endl;
    }		

	double parts_time[mfh.vec.size()];
    nnz = 0;
	for(uint32_t i=0; i<mfh.vec.size(); i++){
		parts_time[i] = 0;
        nnz += mfh.vec[i]->nonZeroes();
	}
		
    T* v = thrust::raw_pointer_cast(&dv[0][0]);
	T* r = thrust::raw_pointer_cast(&dv[1][0]);

    Timer overall, part;
    overall.start();
	for(uint32_t i=0; i<times; i++){
		uint32_t cur_row = 0;
		for(uint32_t j=0; j<mfh.vec.size(); j++){
            if (DEBUG) part.start();
    		
            mfh.vec[j]->multiply(v, r+cur_row);
			cur_row += mfh.vec[j]->get_num_rows();

            if (DEBUG) {
    			cudaThreadSynchronize();
		    	parts_time[j] += part.stop();
                checkCUDAError("kernel call");
            }
		}
        cudaThreadSynchronize();
        checkCUDAError("kernel call");
	}
    double totalTime = overall.stop(); 

	cerr << "Expected/Actual memory usage: " << (total + 2 * hv.size() * sizeof(T) / 1024.0 / 1024.0) << " / " << (dvTotal-free)/1024.0/1024.0 << " MB" << endl;
	stringstream stmp;
	stmp << sizeof(T)*8 << "b," << times <<"iter";
	cerr << setw(15) << stmp.str() 
		 << setw(15) << "seconds" 
		 << setw(15) << "nnz" 
		 << setw(15) << "GFL/s" 
		 << setw(15) << "MB" 
		 << setw(15) << "B/nnz" 
		 << endl;

    if (DEBUG)	
	for(uint32_t j=0; j<mfh.vec.size(); j++){
		cerr << setw(15) << (formats[j] + ":") 
			 << setw(15) << parts_time[j] 
			 << setw(15) << mfh.vec[j]->nonZeroes() 
			 << setw(15) << 2*mfh.vec[j]->nonZeroes()/(parts_time[j]*1000000000)*times 
			 << setw(15) << mfh.vec[j]->memoryUsage()/1024.0/1024.0 
			 << setw(15) << mfh.vec[j]->memoryUsage()/(double)mfh.vec[j]->nonZeroes() 
			 << endl;
	}

	cerr << setw(15) << "Total:"
		 << setw(15) << totalTime 
		 << setw(15) << nnz 
		 << setw(15) << nnz*2/(totalTime*1000000000)*times 
		 << setw(15) << total 
		 << setw(15) << total*1024*1024/nnz 
		 << endl;
	
	thrust::device_vector<T> *dr = &dv[1];
    if (times == 0) dr = &dv[0];

	hv.assign(dr->begin(), dr->end());
	if(print){
		for(uint32_t i=0; i<hv.size(); i++){
			cout << hv[revperm[i]] << endl;
		}
	}
	
}

void readInfo() {
    char infopath[255];

    sprintf(infopath, FORMAT_PERM, matrix_path.c_str());    
    ifstream ifs(infopath);
    DataInputStream permif( ifs );
    permif.readVector( perm );
    uint32_t maxdim = perm.size();
//    for (int i=0;i<maxdim;i++) perm[i] = i;
    revperm.resize(maxdim);
    for (uint32_t i=0;i<maxdim;i++) {
        revperm[perm[i]] = i;
    }
    ifs.close();

    sprintf(infopath , FORMAT_OUTPUT, matrix_path.c_str());
    if (DEBUG) cerr << infopath << endl;
    ifs.open(infopath);
    int nformats;
    ifs >> nformats;
    if (DEBUG) cerr << "Read " << nformats << endl;
    formats.resize(nformats);
    start_rows.resize(nformats);

    for (int i=0;i<nformats;i++) { 
        ifs >> formats[i];
        cerr << formats[i] << " ";
    }
    cerr << "|| " ;
    for (int i=0;i<nformats;i++) {
        ifs >> start_rows[i];
        cerr << start_rows[i] << " ";
    }
    cerr << endl;
     
    ifs.close();
}

void print_usage(char **argv) {
    cout << "Usage: " << argv[0] << " -m <matrix> -b <blocksize> -n <niteration> [ -r (recache) -w (debug) -s > output ]" << endl;
}


int main(int argc, char* argv[]){
    try {
        Params param(argc, argv);
        app.init(param);

        param.getParam("times", &times, true);
        param.getParam("print", &print, true);

		srand(seed);

		if(!app.wdir.empty()){
			cerr << "Changing working directory to " << app.wdir << endl;
			int r = chdir(app.wdir.c_str());
		}
        if (app.datatype == "float")
            execute<float>();
        else
            execute<double>();
				
	} catch(std::exception& e) {
        cerr << e.what() << endl;
        return 1;
    }  
	
	return 0;
}
