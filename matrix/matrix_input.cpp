#ifndef __MATRIX_INPUT_H__
#define __MATRIX_INPUT_H__

#include <string.h>
#include <stdexcept>
#include <iostream>
#include <cstdlib>
#include <algorithm>

#include "matrix_input.h"

using namespace std;

MatrixInput::~MatrixInput(){
}

bool sort_data (std::vector<std::pair< uint32_t , double > > x, std::vector<std::pair< uint32_t, double > > y) { 
    return (x.size() > y.size()); 
}

MMFormatInput::MMFormatInput(std::ifstream &in, std::vector<uint32_t> &permIn) : MatrixInput(), in(in), curRow(0), curIndex(-1), already_read(false), perm(permIn) { 
    //read header only
    char buf[500];

    isBinary = false;
    in.getline(buf, 500);
    if ( strstr(buf, "pattern") != NULL ) {
        isBinary = true; }

    do {
        in.getline(buf, 500);
        if (in.gcount() >=500){ 
            cerr << "FAILED FORMAT" << endl;
            exit(1); 
        }
    } while (buf[0] == '%');

    uint32_t nr, nc, nnz;

    sscanf( buf , "%d %d %d", &nr, &nc, &nnz);

    this->nnz = nnz;
    this->numRows = nr;
    this->numCols = nc;

    if (perm.size() == 0) {
        perm.resize( max(numRows, numCols) );
        for (int i=0; i<perm.size(); i++) {
            perm[i] = i;
        }
    }
}

void MMFormatInput::readIt(){
    in.clear();
    in.seekg(0, std::ios::beg) ;
    cusp::io::read_matrix_market_stream(data, in);
    already_read = true;
}

MatrixInput& MMFormatInput::operator>>(uint32_t &output) {
    if (!already_read) readIt();

    if(curIndex<0){
        output = getNz(perm[curRow]);
        curIndex++;
    } else if(curIndex>=getNz(perm[curRow])){
        curRow++;
        curIndex = -1;
        return (*this >> output);
    } else{
        output = data.column_indices[ data.row_offsets[ perm[curRow] ] + curIndex];
    }
    return *this;
}

MatrixInput& MMFormatInput::operator>>(double &output) {
    output = data.values[ data.row_offsets[ perm[curRow] ] + curIndex];
    curIndex++;
    return *this;
}

#if 0
int main(int argc, char **args){

    if (argc < 2) {
        cout << "Please provide file name" << endl;
        return -1;
    }
    char *filename = args[1];

    ifstream fst(filename);
    vector<uint32_t> perm;
    MMFormatInput f( fst , perm ) ;
    perm[0] = 2;
    perm[1] = 1;
    perm[2] = 0;
    perm[3] = 3;

    int n_rows = f.getNumRows();
    cerr << n_rows << endl;

    uint32_t total = 0;
    for (int i=0;i<n_rows;i++){ 
        uint32_t length;
        f >> length;
        cout << length;
        for (int j=0; j<length; j++) {
            uint32_t c;
            double value;
            f >> c;
            value = f.getValue();
            total++;
            cout << " " << c;
        }
        cout << endl;
    }
    cerr << f.getNNz() << endl;
    cerr << total << endl;

    if (f.getNNz() == total) 
        cerr << " CORRECT " << endl;
    else
        cerr << " WRONG " << endl;

    return 0;
}
#endif
#endif
