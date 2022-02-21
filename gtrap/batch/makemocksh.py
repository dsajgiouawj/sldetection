import numpy as np
import os
import pandas as pd


python_execute="/home/kawahara/gtrap/examples/gtls_slctess.py"
dirname="/home/kawahara/gtrap/examples/sh"
allname="mockall_slc.sh"
eachname="mockeach_slc"


def make_mockall_sh(igtrap,seq=True):
    Nscc=len(igtrap)
    
    f=open(os.path.join(dirname,allname),"w")
    f.write("for i in `seq 0 "+str(Nscc-1)+"`"+"\n")
    f.write("do"+"\n")
    f.write('echo "--------------------------";'+"\n")
    f.write('echo "$i";'+"\n")
    if seq==True:
        f.write(""+os.path.join(dirname,eachname)+'"$i".sh '+"\n")
    else:
        f.write(""+os.path.join(dirname,eachname)+'"$i".sh &> log"$i" &'+"\n")
    f.write('echo "==========================";'+"\n")

    f.write("done"+"\n")
    f.close()      

def make_mock_each(Nbatch, igtrap, Nrep=1,inject=False):
    Nscc=len(igtrap)
    iarr=igtrap["i"].values
#    jarr=igtrap["j"].values
    
    for k in range(0,Nscc):
        filename=eachname+str(k)+".sh"
        f=open(os.path.join(dirname,filename),"w")        
        ender=" & "
        for l in range(0,Nrep):
            i = iarr[k]+l*Nbatch
            j = i + Nbatch - 1

            f.write('i='+str(i)+';'+"\n")
            f.write('j='+str(j)+';'+"\n")
            f.write('echo "$i $j ";'+"\n")
            if l==Nrep-1:
                ender=""
            if Nrep > 1:
                cont=" -cb "+str(l)
            else:
                cont=""
                
            if inject:
                f.write('python '+python_execute+' -i $i -j $j -fig -n 1 -sd "'+str(k)+'" '+cont+ender+"\n")
            else:
                f.write('python '+python_execute+' -i $i -j $j -fig -n 1 -q -sd "'+str(k)+'" '+cont+ender+"\n")
            
        f.close()


        
def make_backend():
    #generate matplotlibrc in sh directory...
    f=open(os.path.join(dirname,"matplotlibrc"),"w")        
    f.write("backend:agg")
    f.close()

def get_h5amount():
    igtrap=pd.read_csv("/home/kawahara/gtrap/data/ctl.list/igtrap.list",delimiter=",",names=("css","i","j"))

    return igtrap


if __name__ == "__main__":
    Nbatch=32 # bacth num for an execute.
    Nrep=4 #num of repeat
    igtrap=get_h5amount()
    #Nsh = 207 # num of eachshells
    
    make_backend()
#    make_mock_each(Nbatch,igtrap,Nrep=Nrep, inject=False)
    make_mock_each(Nbatch,igtrap,Nrep=Nrep, inject=True)
    make_mockall_sh(igtrap,seq=True)
