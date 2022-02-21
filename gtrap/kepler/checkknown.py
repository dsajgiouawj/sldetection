import numpy as np
import pandas as pd


koi=pd.read_csv("../data/dr25koi.csv",delimiter=",",comment="#")
#print(koi["kepid"])

keb=pd.read_csv("../data/keb.csv",delimiter=",",comment="#")
#print(keb["KIC"])

kelp=pd.read_csv("../data/primary1127.dat",delimiter=",",comment="#")
#print(kelp["KID"])


for ipick in range(0,10):
    check=pd.read_csv("picklc/group"+str(ipick)+".true.dat",names=('KIC',))
    #print(check["KIC"])
    ckoi=[]
    ckeb=[]
    ckelp=[]

    for ikic in check["KIC"]:
#        pick.append(ipick)
        
        mask=koi["kepid"]==ikic    
        a=len(koi["kepid"][mask])
        if a>0 :
            ckoi.append(1)
        else:
            ckoi.append(0)

        mask=keb["KIC"]==ikic    
        a=len(keb["KIC"][mask])
        if a>0 :
            ckeb.append(1)
        else:
            ckeb.append(0)
        
        mask=kelp["KID"]==ikic    
        a=len(kelp["KID"][mask])
        if a>0 :
            ckelp.append(1)
        else:
            ckelp.append(0)
               
    ckoi=np.array(ckoi)
    ckeb=np.array(ckeb)
    ckelp=np.array(ckelp)

    tot=ckoi+ckeb+ckelp
    #print(ckoi+ckeb+ckelp)
    
    mask=tot==0
    #(check[mask].to_csv("unknown.dat"))
    
    for j in check[mask]["KIC"]:
        print("mv pred"+str(j)+"."+str(ipick)+".png try1/unknown")
    
