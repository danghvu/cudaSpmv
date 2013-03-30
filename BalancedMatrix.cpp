
#include "BalancedMatrix.h"

using namespace std;

BalancedMatrix::BalancedMatrix(csr_matrix &matrixIn, vector<uint32_t> &oldpermIn, uint32_t startRowNoIn, uint32_t numRowsIn, uint32_t numProcsIn, uint32_t numRowsPerSlice) :
oldperm(oldpermIn), matrix(matrixIn), startRowNo(startRowNoIn), numRows(numRowsIn), numProcs(numProcsIn){

	if(numRows % (numProcs*numRowsPerSlice) != 0){
		cout << "Number of rows is not divisible by the number of multi-processor, will not do balancing" << endl;
		permutation.reserve(numRows);
		for(uint32_t i=0; i<numRows; i++){
			permutation.push_back(oldperm[i+startRowNo]);
		}
	} else{
		balance();
	}
}

void BalancedMatrix::balance(){
	vector<Slice> vec(numProcs);
	priority_queue<Slice*, vector<Slice*>, Slice > pq;
    
    for(uint32_t i=0; i<numProcs; i++){
	    pq.push(&vec[i]);
    }

	for(uint32_t i=0; i<numRows; i++){
        Slice* sc = pq.top();
		pq.pop();
		sc->addRow(oldperm[i+startRowNo], matrix.row_offsets[ oldperm[i+startRowNo] + 1] - matrix.row_offsets[ oldperm[i+startRowNo] ]);
		if(sc->num_rows < numRows/numProcs){
			pq.push(sc);
		}
	}

	permutation.reserve(numRows);
	for(uint32_t i=0; i<numProcs; i++){
		permutation.insert(permutation.end(), vec[i].row_start.begin(), vec[i].row_start.end());
	}
}

//int main(int argc, char** argv){
//	const int NUMROWS=1600;
//	const int NUMPROCS=16;
//	const int startRow = 0;
//	vector<vector<uint32_t> > matrix(NUMROWS);
//	for(uint32_t i=0; i<NUMROWS; i++){
//		matrix[i].assign(NUMROWS-i, 0);
//	}
//	BalancedMatrix bm(matrix, startRow, NUMROWS-startRow, NUMPROCS);
//	uint32_t sum=0;
//	vector<bool> flag(NUMROWS);
//	for(uint32_t i=0; i<NUMROWS-startRow; i++){
//		sum+=matrix[bm.permutation[i]].size();
//		if(bm.permutation[i]>=NUMROWS || bm.permutation[i]<startRow){
//			cerr << "Invalid row!" << enl;
//		} else if(flag[bm.permutation[i]]){
//			cerr << "Duplicate row!" << endl;
//			break;
//		} else{
//			flag[bm.permutation[i]] = true;
//		}
//		if(((i+1)%((NUMROWS-startRow)/NUMPROCS)) == 0){
//			cout << sum << endl;
//			sum = 0;
//		}
//	}
//}
