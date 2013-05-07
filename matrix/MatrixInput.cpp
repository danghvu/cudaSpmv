#include "MatrixInput.h"
#include <string.h>
#include <stdexcept>
#include <iostream>
#include <cstdlib>
#include <algorithm>
using namespace std;

MatrixInput::~MatrixInput(){
}

FileMatrixInput::FileMatrixInput(std::ifstream &in) : MatrixInput(), in(in){
}

bool sort_data (std::vector<std::pair< uint32_t , double > > x, std::vector<std::pair< uint32_t, double > > y) { 
    return (x.size() > y.size()); 
}

//--------------------------------------------------------------------------------------------------


//--------------------------------------------------------------------------------------------------

MMFormatInput::MMFormatInput(std::ifstream &in, std::vector<uint32_t> &permIn) : MatrixInput(), in(in), curRow(0), curIndex(-1), already_read(false), perm(permIn) { 
    
    //read header only
    char buf[500];
    
    isBinary = false;
    in.getline(buf, 500);
    if ( strstr(buf, "pattern") != NULL ) {
        isBinary = true;
    }
    
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
		output = data.row_offsets[perm[curRow] + 1] - data.row_offsets[perm[curRow]];
		curIndex++;
	} else if(curIndex>=data.row_offsets[perm[curRow] + 1] - data.row_offsets[perm[curRow]]){
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

MatrixInput& FileMatrixInput::operator>>(uint32_t &output){
	in >> output;
	return *this;
}

VectorMatrixInput::VectorMatrixInput(std::vector<uint32_t> &vec, std::vector<uint32_t> &rowStart, uint32_t curRow) :
		MatrixInput(), vec(vec), rowStart(rowStart), curRow(curRow), curIndex(0), remainingNnz(0){
}

MatrixInput& VectorMatrixInput::operator>>(uint32_t &output){
	if(remainingNnz == 0){
		if(curRow < rowStart.size()){
			curIndex = rowStart[curRow];
			remainingNnz = output = vec[curIndex++];
		} else{
			throw std::ios_base::failure("End of row reached.");
		}
	} else{
		output = vec[curIndex++];
		remainingNnz--;
	}

	if(remainingNnz == 0){
		curRow++;
	}

    return *this;
}

//------------------------------------------------------------------------------------------------

VectorOfVectorMatrixInput::VectorOfVectorMatrixInput(csr_matrix &vecIn, vector<uint32_t> &permutationIn) :
		vec(vecIn), perm(permutationIn), curIndex(-1), curRow(0){
}

VectorOfVectorMatrixInput::VectorOfVectorMatrixInput(csr_matrix &vecIn, vector<uint32_t> &perm, uint32_t numRows, uint32_t curRowIn) :
		vec(vecIn), perm(perm), curIndex(-1), curRow(0){
	//for(uint32_t i=0; i<numRows; i++){
	//	permutation[i] = curRowIn+i;
	//}
}

MatrixInput& VectorOfVectorMatrixInput::operator >>(double & output) {
//    output = vec[permutation[curRow]][curIndex].second;
    output = vec.values[ vec.row_offsets[ perm[curRow] ] + curIndex];
    curIndex++;
    return *this;
}

MatrixInput& VectorOfVectorMatrixInput::operator >>(uint32_t& output){
	if(curRow>=perm.size()){
		throw std::ios_base::failure("End of row reached");
	}
    if(curIndex<0){
		output = vec.row_offsets[perm[curRow] + 1] - vec.row_offsets[perm[curRow]];
		curIndex++;
	} else if(curIndex>=vec.row_offsets[perm[curRow] + 1] - vec.row_offsets[perm[curRow]]){
		curRow++;
		curIndex = -1;
		return (*this >> output);
	} else{
		output = vec.column_indices[ vec.row_offsets[ perm[curRow] ] + curIndex];
	}
	return *this;
/*
	if(curIndex<0){
		output = vec[permutation[curRow]].size();
		curIndex++;
	} else if(curIndex>=vec[permutation[curRow]].size()){
		curRow++;
		curIndex = -1;
		return (*this >> output);
	} else{
		output = vec[permutation[curRow]][curIndex].first;
//		curIndex++;
	}
	return *this; */
}

int __main(int argc, char **args){
    
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

