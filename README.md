cudaSpmv
========
CUDA Sparse-Matrix Vector Multiplication using the Sliced Coordinate format

`prep`: to partition the input matrix, store the partitioned matrix in binary for faster access

`spmv`: perform spmv with prepared files from `prep`

`cusp`: SpMV using cusp library matrix formats

Usage examples:
------------------

    ./prep matrix=./samples/test.mtx datatype=float
    ./spmv matrix=./samples/test.mtx datatype=float vector=./samples/vector.txt outfile=/tmp/output 
    
Parameters:
---------------------
Required:

 - `matrix`: path to the input matrix file stored in Matrix Market Format
 - `datatype`: `float` or `double`

Optional:

 - `vector`: path to the input vector, default is a unit vector
 - `outfile`: output file for `spmv`, will not output by default
 - `wdir`: working directory (all temporary files and caches are stored here), default is the current directory
 - `gpu`: id of the GPU to execute, default is `0`
 - `times`: number of SpMV iterations, default is `1`, if more than `1` `sortcol` should be enabled to ensure the same result at the end
 - `sortcol`: `0|1` set as `1` to permute columns the same as rows (only useful in `prep`), default is 0
 - `sortrow`: `0|1` set as `1` force sorting rows, default is 0, however will still sort if rows weights differ by *a lot* (only useful in `prep`)

