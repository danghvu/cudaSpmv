file = File.open('/home/dang/list').readlines()

file.each {|mat|

matrix = mat.match(/([^\.\/]+\.mtx)/)[1]
folder = "/home/dang/data/profile_3.0/"

$shared = ['profile_shared64']

sh_load = {}
sh_store = {}
ins1 = {}
ins2 = {}

$shared.each_with_index { |share,i|
    #inst = $inst[i]

    fshare = "#{folder}#{share}/#{matrix}.profile"
    
    def read_profile(f, h1, h2, h3, h4)
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
        
                if not h1.include?a[0] then 
                    h1[a[0]] = 0   
                    h2[a[0]] = 0
                    h3[a[0]] = 0
                    h4[a[0]] = 0
                end
                h1[a[0]]+= a[4].to_i
                h2[a[0]]+= a[5].to_i
                h3[a[0]]+= a[6].to_i
                h4[a[0]]+= a[7].to_i
            end
        }
    end

    read_profile(fshare, sh_load, sh_store, ins1, ins2)
}

File.open("result.csv",'a') {|f|
    f.print "#{matrix}, "
    
    sh_load.each { |k,v|
        l = 100 * (v + sh_store[k]).to_f / (ins1[k] + ins2[k]).to_f
        f.print "#{k}, #{l}"
    }
    f.puts
}

}

