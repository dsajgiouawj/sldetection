import numpy as np
import os
Nbatch=50 # bacth num for an execute.
Neach = 1000 # num for each
Nsh = 124 # num of eachshells
# total light curves = Nbatch*Neach*Nsh

python_execute="../gtls_slctess.py"
dirname="/home/kawahara/gtrap/examples/sh"
allname="pickall_slc.sh"
eachname="pickeach_slc"


def make_pickall_sh():
    #execute this sh for all shells
    
    f=open(os.path.join(dirname,allname),"w")
    f.write("for i in `seq 1 "+str(Nsh)+"`"+"\n")
    f.write("do"+"\n")
    f.write('echo "$i";'+"\n")
    f.write(os.path.join(dirname,eachname)+'"$i".sh &> log"$i" &'+"\n")
    f.write("done"+"\n")
    f.close()      

def make_pick_each():
    
    for i in range(1,Nsh+1):
        filename=eachname+str(i)+".sh"
        f=open(os.path.join(dirname,filename),"w")        
        ex = Neach*(i)+1
        fx = Neach*(i+1)
        f.write('for i in `seq '+str(ex)+' '+str(fx)+'`'+"\n")
        f.write('do'+"\n")
        f.write('a='+str(Nbatch)+';'+"\n")
        f.write('s=$(($a * $i));'+"\n")
        f.write('e=$(($s + $a - 1));'+"\n")
        f.write('echo "$i $s $e";'+"\n")
        f.write('python '+python_execute+' -i $s -j $e -n 2 -p -sd "'+str(i)+'";'+"\n")
        f.write('done'+"\n")
        f.close()
    
if __name__ == "__main__":
    make_pick_each()
    make_pickall_sh()
