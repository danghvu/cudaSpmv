
File.open(ARGV[0],'r').each_line do |line|
    if line =~ /benching \.\.\. .+\/([^\/]+\.mtx)/ then
        puts
        print $1,', '
    end
    if line =~ /Total:\s+\S+\s+\S+\s+(\S+)/ then
        print $1,', '
    end 
    if line =~ /CSR:.+ (\S+) GFL/ then
        print $1,', '
    end
    if line =~ /COO:.+ (\S+) GFL/ then
        print $1,', '
    end
    if line =~ /HYB:.+ (\S+) GFL/ then
        print $1,', '
    end
end
puts
'''
CSR: 0.139417(s) ---> 8.6223 GFL/s
 SIZE: 91
COO: 0.189174(s) ---> 6.35445 GFL/s
 SIZE: 77
HYB: 0.0862471(s) ---> 13.9378 GFL/s
'''
