/* vim: set ft=cpp: */
#include <stdint.h>
#include <iostream>
#include <fstream>
#include <algorithm>

#include "util/data_input_stream.h"
#include "util/data_output_stream.h"
#include "util/timer.h"
#include "util/parameters.h"
#include "matrix/matrix_input.h"
#include "partition.h"

using namespace std;
Parameter app;

string filename;
int blocksize = 32;
bool verbose = false, sortcol = false, sortrow = false;

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

    
    double theta;
    if (mean < median) theta = std::min(median, mean * 2); //combine more row
    else theta = std::max(median, mean / 2); // combine less row
    // if small variance do not sort
    if (max - min > 256 || sortrow)  {
        cout << "Decide to sort!" << endl;
    } else {
        for (int i=0; i<maxSize; i++) f.perm[i] = i;
    }

    cout << "Max: " << max << ", Min:" << min << ", Median: " << median << ", Mean:" << mean << ", Std:" << std << ", Theta: " << theta << endl;

    Timer timer;
    timer.start();
    CutOffFinder<T> cof(f, sortcol);
    cof.theta = theta;
    cof.execute(verbose);
    cout << "Spending " << timer.stop() << "s in finding cutoff" << endl;
    cof.printResult(filename);
    cof.writeCache(filename);
}

int main(int argc, char* argv[]){
    try {
        Params param(argc, argv);
        app.init(param);

        param.getParamOptional("sortcol", &sortcol);
        param.getParamOptional("sortrow", &sortrow);

		if(!app.wdir.empty()){
			cout << "Changing working directory to " << app.wdir << endl;
			int r = chdir(app.wdir.c_str());
		}
        filename = getFileName(app.matrixFile);
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
