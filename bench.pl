#!/usr/bin/perl

use strict;
use warnings;

use constant { 
    SCOO_EXEC => "/home/danghvu/workspace/CudaSPMV/cuda_bwc",
    PREP_EXEC => "/home/danghvu/workspace/CudaSPMV/prep",
    CUSP_EXEC => "/home/danghvu/workspace/cusp-bench/benchcusp",
    NTIMES => 100,
    BENCH => { 'scoo' => 1, 'cusp' => 1 }
};

my $matrix_file = shift;
my $gpu = shift;

$gpu = "2" if (!$gpu);

die("Please input matrix name") if (!$matrix_file);

print "start benching ... " . $matrix_file . "\n";

die("Can't find $matrix_file") if (!-f $matrix_file);

if (BENCH->{'scoo'} == 1) {
    print"PREP...\n";
    if (! -f "$matrix_file.prep") {
        system( PREP_EXEC . " $matrix_file ") == 0
            or die("$?");
        system( "touch $matrix_file.prep");
    }
    print "SCOO...\n";
    system( SCOO_EXEC . " -g $gpu -b 32 -m $matrix_file -n " . NTIMES ) == 0 
        or die("$?");
}

if (BENCH->{'cusp'} == 1) {
    print "CUSP...\n";
    system( CUSP_EXEC . " -g $gpu -b 32 -m $matrix_file -n ". NTIMES ) == 0
        or die("$?");
}

print "DONE!\n";
