#!/bin/sh
make
./prep matrix=samples/test.mtx datatype=float
./spmv matrix=samples/test.mtx datatype=float outfile=test.mtx.out.txt vector=samples/vector.txt
