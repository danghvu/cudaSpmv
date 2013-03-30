#ifndef BALANCEDMATRIX_H_
#define BALANCEDMATRIX_H_

#include "matrix/MatrixInput.h"
#include <vector>
#include <stdint.h>

#include <queue>
#include <iostream>
#include <stdexcept>

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


class BalancedMatrix {

public:
	csr_matrix &matrix;
	std::vector<uint32_t> permutation;
    std::vector<uint32_t> &oldperm;
	uint32_t startRowNo;
	uint32_t numRows;
	uint32_t numProcs;
    
    int32_t lastStartRowNo;
    int32_t lastNumRows;

	std::vector<Slice> vec;
	std::priority_queue<Slice*, std::vector<Slice*>, Slice > pq;

	BalancedMatrix(csr_matrix &matrixIn, std::vector<uint32_t> &permIn, uint32_t startRowNoIn, uint32_t numRowsIn, uint32_t numProcsIn, uint32_t numRowsPerSlice);

private:
	void balance();
};

#endif /* BALANCEDMATRIX_H_ */
