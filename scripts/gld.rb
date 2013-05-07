
file = File.open('/home/dang/list').readlines()

file.each {|mat|

matrix = mat.match(/([^\.\/]+\.mtx)/)[1]
folder = "/home/dang/data/profile_2.0/"

$gld = ['profile_global_load']

gl32 = {}
gl16 = {}
gpu = {}

$gld.each_with_index { |gld,i|
    fgld = "#{folder}#{gld}/#{matrix}.profile"
    fgldc = "#{folder}#{gld}/#{matrix}.cusp.profile"
    
    def read_profile(f, gl32, gl16, gpu)
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
        
                if not gl32.include?a[0] then 
                    gl32[a[0]] = 0 
                    gl16[a[0]] = 0
                    gpu[a[0]] = 0
                end
                gpu[a[0]]+= a[1].to_f
                gl32[a[0]]+= a[4].to_i
                gl16[a[0]]+= a[5].to_i
            end
        }
    end

    read_profile(fgld, gl32, gl16, gpu)
    read_profile(fgldc, gl32, gl16, gpu)
}

File.open("result.csv",'a') {|f|
    f.print "#{matrix}, "
    
    gl32.each { |k,v|
        l = (gl32[k]*4 + gl16[k]*2) / (gpu[k]/(10**6))
        f.print "#{k}, #{l/1024/1024/1024}," #{ld_hit}, " #{tex_tp}, "
    }
    f.puts
}

}

