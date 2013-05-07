#!/bin/sh
make
./prep matrix=samples/test.mtx datatype=float sortrow=1 sortcol=1 wdir=./
./spmv matrix=samples/test.mtx datatype=float outfile=test.mtx.out.txt vector=samples/vector.txt times=10 wdir=./
./cusp -m samples/test.mtx -v samples/vector.txt -b 32 -n 10 -s > test.mtx.cusp.out.txt
if diff test.mtx.cusp.out.txt test.mtx.out.txt > /tmp/diff.test; then
    echo "CORRECT";
else
    echo "INCORRECT";
    cat /tmp/diff.test;
fi
rm test.mtx*
