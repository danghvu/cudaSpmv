/*
 * MatrixInput.h
 *
 *  Created on: May 12, 2011
 *      Author: hans
 */

#ifndef MATRIXINPUT_H_
#define MATRIXINPUT_H_

#include <stdint.h>
#include <fstream>
#include <vector>
#include <cusp/io/matrix_market.h>
#include <cusp/coo_matrix.h>


const static char * FORMAT_OUTPUT = "%s.info";
const static char * FORMAT_PERM   = "%s.perm.bin";

class MatrixInput{
public:
	virtual ~MatrixInput() = 0;
	virtual MatrixInput& operator>>(uint32_t &output) = 0;
    virtual double getValue() { return 1; };

protected:
    bool isBinary;

};

class FileMatrixInput : public MatrixInput{
	std::ifstream &in;
public:
	FileMatrixInput(std::ifstream &in);
	MatrixInput& operator>>(uint32_t &output);
};

typedef std::vector< std::vector< std::pair< uint32_t , double > > > vectorOfvectorDouble;
typedef cusp::csr_matrix<uint32_t, double, cusp::host_memory> csr_matrix;


static int getCSRNumRows( const csr_matrix & c, int row_number ) {
    return c.row_offsets[row_number+1] - c.row_offsets[row_number];
}

class MMFormatInput : public MatrixInput {
    private:
        std::ifstream &in;
        uint32_t nnz, numRows, numCols; //temporary for header only

    protected:
        uint32_t curRow;        
        int curIndex;
        bool already_read;

    public:
        csr_matrix data;
        std::vector<uint32_t> &perm;
        uint32_t getNumRows() { return numRows; }
        uint32_t getNumCols() { return numCols; }
        uint32_t getNNz() { return nnz; }  

        MMFormatInput(std::ifstream &in, std::vector<uint32_t> &perm);
        MatrixInput& operator >>(uint32_t &output);
        MatrixInput& operator >>(double &output);

        uint32_t getNz(int r) {
            return data.row_offsets[r+1] - data.row_offsets[r];
        }

        void readIt();

        double getValue() { double v; *this >> v; return v; }
};

class VectorMatrixInput : public MatrixInput{
	std::vector<uint32_t> &vec;
	std::vector<uint32_t> &rowStart;
	uint32_t curIndex;
	uint32_t curRow;
	uint32_t remainingNnz;

public:
	VectorMatrixInput(std::vector<uint32_t> &vec, std::vector<uint32_t> &rowStart, uint32_t curRow);
	MatrixInput& operator>>(uint32_t &output);
};


struct VectorOfVectorMatrixInput : public MatrixInput{
	csr_matrix &vec;
	std::vector<uint32_t> perm;
	int curIndex;
	uint32_t curRow;

public:
	VectorOfVectorMatrixInput(csr_matrix &vecIn, std::vector<uint32_t> &permutationIn);
	VectorOfVectorMatrixInput(csr_matrix &vecIn, std::vector<uint32_t> &perm, uint32_t numRows, uint32_t curRowIn);
	MatrixInput& operator>>(uint32_t &output);
	MatrixInput& operator>>(double &output);
    double getValue() { double v; *this >> v; return v; }

};

#endif /* MATRIXINPUT_H_ */
