cudaSpmv
========
CUDA Sparse-Matrix Vector Multiplication using the Sliced Coordinate format

`prep`: to partition matrix, store result in binary for faster access
`spmv`: perform spmv with prepared files from `prep`

Sample usages: 
------------------

    ./prep matrix=./samples/test.mtx datatype=float
    ./spmv matrix=./samples/test.mtx datatype=float vector=./samples/vector.txt outfile=/tmp/output 
    
Parameters:
---------------------
Required:

 - `matrix`: path to matrix file in Matrix Market Format
 - `datatype`: `float` or `double`

Optional:

 - `vector`: input vector, default is a unit vector
 - `outfile`: output file for `spmv`, will not output by default
 - `wdir`: working directory (all temprary files and caches are stored here), default is the current directory
 - `gpu`: id of the GPU to execute, default is `0`
 - `times`: number of SpMV iteration, default is `1`, if more than `1` enable `sortcol` to ensure the same result at the end
 - `sortcol`: `0|1` sorting the column as the rows (only useful in `prep`), default is 0
 - `sortrow`: `0|1` force to sort the row, default is 0, will still sort if rows weights differ *a lot* (only useful in `prep`)

