
file = File.open('/home/dang/list').readlines()

file.each {|mat|

matrix = mat.match(/([^\.\/]+\.mtx)/)[1]
folder = "/home/dang/data/profile_2.0/"

$shared = ['profile_shared64']
$inst =  ['profile_instissue64']

sh = {}
ins = {}

$shared.each_with_index { |share,i|
    inst = $inst[i]

    fshare = "#{folder}#{share}/#{matrix}.profile"
    fins  = "#{folder}#{inst}/#{matrix}.profile"
    
    def read_profile(f, h)
        onhyb = false 

        File.open(f,'r').each_line { |line|
            if line =~ /kernel/ then
                a = line.split(',')
                if a[0] =~ /sliced_coo/ then
                    a[0] = 'SCOO'
                elsif a[0] =~ /csr/ then
                    a[0] = 'CSR'
                elsif a[0] =~ /coo/ then
                    if onhyb then
                        a[0] = 'HYB'
                    else
                        a[0] = 'COO'
                    end
                elsif a[0] =~ /ell/ then
                    onhyb = true 
                    a[0] = 'HYB'
                end
        
                if not h.include?a[0] then 
                    h[a[0]] = 0   
                end
                h[a[0]]+= a[4].to_i
                #ld1[a[0]]+= a[5].to_i
            end
        }
    end

    read_profile(fshare, sh)
    read_profile(fins, ins)
}

File.open("result.csv",'a') {|f|
    f.print "#{matrix}, "
    
    sh.each { |k,v|
        l = 100 * v.to_f / ins[k].to_f
        f.print "#{k}, #{l}"
    }
    f.puts
}

}

