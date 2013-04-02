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

char *filename;
int blocksize = 32;
bool verbose = false;
bool trial = true;

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

    vector_sort mysort ( &f.data );
    
    sort( f.perm.begin() , f.perm.begin() + f.getNumRows(), mysort );

    if (trial == true) {
        CutOffFinder<T> cof(f);
        Timer timeex;
        timeex.start();
        cof.execute_trial(verbose);
        cerr << "Spending " << timeex.stop() << "s in finding cutoff" << endl;
        cof.printResult(filename);
        cof.writeCache(filename);
    } 
    else  {
        CutOffFinder<T> cof( f );
        Timer timeex;
        timeex.start();
        cof.execute(verbose);
        cerr << "Spending " << timeex.stop() << "s in finding cutoff" << endl;
        cof.printResult(filename);
        cof.writeCache(filename);
    }
}

int main(int argc, char* argv[]){
    try {
        Params param(argc, argv);
        app.init(param);

        int seed = 0;
        param.getParam("seed", &seed, true);
		srand(seed);

		if(!app.wdir.empty()){
			cerr << "Changing working directory to " << app.wdir << endl;
			int r = chdir(app.wdir.c_str());
		}
        filename = (char *)app.matrixFile.c_str();

        ifstream inf(filename);
        vector<uint32_t> tempperm;
        MMFormatInput f(inf, tempperm);
        f.readIt();

        if (app.datatype == "float")
            preprocessing<float>(f);
        else
            preprocessing<double>(f);
				
	} catch(std::exception& e) {
        cerr << e.what() << endl;
        return 1;
    }  
	
	return 0;
}
