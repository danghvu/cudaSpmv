#include <stdint.h>
#include "util/data_input_stream.h"
#include "util/data_output_stream.h"
#include "matrix/MatrixInput.h"
#include "CutOffFinder.h"
#include "util/timer.h"

#include <iostream>
#include <fstream>
#include <algorithm>
#include "parameters.h"

using namespace std;
Parameter app;

string filename;
int seed=0, blocksize = 32;
bool verbose = false, trial = false, sortcol = false, sortrow = false;

struct vector_sort { 
    csr_matrix * data;
    vector_sort( csr_matrix *d ) : data(d) {}
    bool operator () ( int i, int j ) {
        return ( data->row_offsets[i+1] - data->row_offsets[i] > data->row_offsets[j+1] - data->row_offsets[j]); 
    } 
};

template<typename T> 
void preprocessing ( MMFormatInput &f ) {
    
    uint32_t maxSize = max(f.getNumRows(), f.getNumCols());
    f.perm.resize(maxSize, 0);

    for (int i=0; i<maxSize; i++) f.perm[i] = i;
    double mean = 1.0*f.getNNz()/f.getNumRows();

    uint32_t max = f.getNz(0);
    uint32_t min = max;

    double std = (mean-max)*(mean-max);

    for (int i=1; i<f.getNumRows(); i++) {
        uint32_t nz = f.getNz(i);
        if (nz == 0) continue;
        if (nz > max) max = nz;
        if (nz < min || min == 0) min = nz;

        std += (nz-mean)*(nz-mean);
    }
    std = sqrt(std/f.getNumRows());
 
    vector_sort mysort ( &f.data );
    sort( f.perm.begin() , f.perm.begin() + f.getNumRows(), mysort );

    double median = f.getNz(f.perm[f.getNumRows()/2]);

    cout << "Max: " << max << ", Min:" << min << ", Median: " << median << ", Mean:" << mean << ", Std:" << std << endl;

    double alpha;
    if (mean < median) alpha = std::min(median, mean * 2); //combine more row
    else alpha = std::max(median, mean / 2); // combine less row
    // if small variance do not sort
    if (max - min > 256 || sortrow)  {
        cout << "Decide to sort!" << endl;
    } else {
        for (int i=0; i<maxSize; i++) f.perm[i] = i;
    }

    cout << alpha << endl;

    if (trial == true) {
        CutOffFinder<T> cof(f, sortcol);
        Timer timeex;
        timeex.start();
        cof.execute_trial(verbose);
        cout << "Spending " << timeex.stop() << "s in finding cutoff" << endl;
        cof.printResult(filename);
        cof.writeCache(filename);
    } 
    else  {
        CutOffFinder<T> cof(f, sortcol);
        Timer timeex;
        timeex.start();
        cof.alpha = alpha;
        cof.execute(verbose);
        cout << "Spending " << timeex.stop() << "s in finding cutoff" << endl;
        cof.printResult(filename);
        cof.writeCache(filename);
    }
}

int main(int argc, char* argv[]){
    try {
        Params param(argc, argv);
        app.init(param);

        param.getParamOptional("seed", &seed);
		srand(seed);

        param.getParamOptional("sortcol", &sortcol);
        param.getParamOptional("sortrow", &sortrow);

		if(!app.wdir.empty()){
			cout << "Changing working directory to " << app.wdir << endl;
			int r = chdir(app.wdir.c_str());
		}
        filename = getFileName(app.matrixFile);
        cout << filename << endl;

        ifstream inf(app.matrixFile.c_str());
        vector<uint32_t> tempperm;
        MMFormatInput f(inf, tempperm);
        f.readIt();

        if (app.datatype == "float")
            preprocessing<float>(f);
        else {
            preprocessing<double>(f);
        }
				
	} catch(std::exception& e) {
        cerr << e.what() << endl;
        return 1;
    }  
	
	return 0;
}
