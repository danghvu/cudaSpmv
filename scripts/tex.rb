
#file = File.open('/home/dang/list').readlines()
file = ['']

file.each {|mat|

#matrix = mat.match(/([^\.\/]+\.mtx)/)[1]
matrix = ''
folder = "/home/dang/tmp/" #"/home/dang/data/profile_3.0/"

$hit = ['hit']#'profile_tex0_query64', ]#'profile_tex1_hit', 'profile_tex2_hit', 'profile_tex3_hit']
$miss =['miss']#'profile_tex0_miss64', ]#, 'profile_tex1_miss', 'profile_tex2_miss', 'profile_tex3_miss']

tex_hit = {}
gpu_hit = {}
tex_miss = {}
gpu_miss = {}

$hit.each_with_index { |hit,i|
    miss = $miss[i]

    fhit, fmiss = "#{folder}#{hit}#{matrix}.profile", "#{folder}#{miss}#{matrix}.profile"
    #cfhit, cfmiss = "#{folder}#{hit}/#{matrix}.cusp.profile", "#{folder}#{miss}/#{matrix}.cusp.profile"
    
    def read_profile(f, tex, gpu)
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
        
                if not tex.include?a[0] then 
                    tex[a[0]] = 0 
                    #ld1[a[0]] = 0   
                    gpu[a[0]] = 0
                end
                gpu[a[0]]+= a[1].to_f
                tex[a[0]]+= a[4].to_i
                #ld1[a[0]]+= a[5].to_i
            end
        }
    end

    read_profile(fhit, tex_hit, gpu_hit)
    read_profile(fmiss, tex_miss, gpu_miss)
    #read_profile(cfhit, tex_hit, gpu_hit)
    #read_profile(cfmiss, tex_miss, gpu_miss)
}

File.open("result.csv",'a') {|f|
    f.print "#{matrix}, "
    
    tex_hit.each { |k,v|
    #    f.puts "#{k}, #{v}, #{tex_miss[k]}, #{ld1_hit[k]}, #{ld1_miss[k]}"
    #    f.print "#{v}, #{tex_miss[k]}, #{ld1_hit[k]}, #{ld1_miss[k]}, "
        tex_hit = [(v-tex_miss[k]).to_f / v * 100, 0].max
        #ld_hit = [ld1_hit[k].to_f / (ld1_hit[k] + ld1_miss[k]) * 100, 0].max
        #tex_tp = v.to_f*32 / gpu_hit[k]
        f.print "#{k}, #{tex_hit}," #{ld_hit}, " #{tex_tp}, "
    }
    f.puts
}

}

