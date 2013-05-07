#!/usr/bin/perl

use strict;
use warnings;

use constant { 
    SCOO_EXEC => "/home/dang/cudaSpmv/spmv",
    PREP_EXEC => "/home/dang/cudaSpmv/prep",
    CUSP_EXEC => "/home/dang/cusp-bench/cusp",
    NTIMES => 100,
    BENCH => { 'scoo' => 1, 'cusp' => 0 }
};

sub _system{
    my $s = shift;
    print STDERR $s . "\n";
    return system($s);
}

my $matrix_file = shift;
my $gpu = shift;
my $type = 'float';

$gpu = "0" if (!$gpu);

die("Please input matrix name") if (!$matrix_file);

print "start benching ... " . $matrix_file . "\n";
print STDERR "start benching ... " . $matrix_file . "\n";

die("Can't find $matrix_file") if (!-f $matrix_file);

my $common = "matrix=$matrix_file wdir=/home/dang/data/tmp/ datatype=$type sortcol=1 gpu=$gpu times=".NTIMES;

$ENV{'CUDA_PROFILE_LOG'} = "$matrix_file.profile";

if (BENCH->{'scoo'} == 1) {
    print"PREP...\n";
    if (! -f "$matrix_file.prep.$type") {
        _system( PREP_EXEC . " $common") == 0
            or die("$?");
        _system( "touch $matrix_file.prep.$type");
    }
    print "SCOO...\n";
    _system(SCOO_EXEC . " $common") == 0 
        or die("$?");
}

$ENV{'CUDA_PROFILE_LOG'} = "$matrix_file.cusp.profile";

if (BENCH->{'cusp'} == 1) {
    print "CUSP...\n";
    _system( CUSP_EXEC . " -g $gpu -b 64 -m $matrix_file -n ". NTIMES ) == 0
        or die("$?");
}

print "DONE!\n";
