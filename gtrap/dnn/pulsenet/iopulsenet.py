import numpy as np


def makearrCA(flist,default_pick=0):
    #LC and local BIP and local AST
    #,[lab],lcs,lcsw,asinds,asindsw,infogap,infogapw,starinfo
    lab=[]
    X=[]
    Xw=[]
    B=[]
    A=[]

    info=[]
    for fn in flist:
        d=np.load(fn,allow_pickle=True)
        if d["arr_0"][0]==-1:
            labtmp=default_pick
        else:
            labtmp=d["arr_0"][0]
        lab.append(labtmp)
        X.append(np.array([d["arr_1"]+1.0]).T)
        Xw.append(np.array([d["arr_2"]+1.0]).T)
        B.append(np.array([d["arr_7"]+1.0,d["arr_9"]+1.0]).T)
        A.append(np.array([d["arr_4"]+1.0]).T)
        info.append(d["arr_11"])
    lab=np.array(lab).astype(np.int32)
    X=np.array(X)
    Xw=np.array(Xw)
    B=np.array(B)
    A=np.array(A)

    info=np.array(info)
    
    return lab,X,Xw,B,A,info




def makearrCS(flist,default_pick=0):
    #LC and BIP
    #,[lab],lcs,lcsw,asinds,asindsw,infogap,infogapw,starinfo
    lab=[]
    X=[]
    Xw=[]
    G=[]
    info=[]
    for fn in flist:
        d=np.load(fn,allow_pickle=True)
        if d["arr_0"][0]==-1:
            labtmp=default_pick
        else:
            labtmp=d["arr_0"][0]
        lab.append(labtmp)
        X.append(np.array([d["arr_1"]+1.0]).T)
        Xw.append(np.array([d["arr_2"]+1.0]).T)
        G.append(np.array([d["arr_7"]+1.0,d["arr_9"]+1.0]).T)
        info.append(d["arr_11"])
    lab=np.array(lab).astype(np.int32)
    X=np.array(X)
    Xw=np.array(Xw)
    G=np.array(G)
    info=np.array(info)
    
    return lab,X,Xw,G,info


def makearrCL(flist,default_pick=0):
    #LC and GAP
    #,[lab],lcs,lcsw,asinds,asindsw,infogap,infogapw,starinfo
    lab=[]
    X=[]
    Xw=[]
    G=[]
    info=[]
    for fn in flist:
        d=np.load(fn,allow_pickle=True)
        if d["arr_0"][0]==-1:
            labtmp=default_pick
        else:
            labtmp=d["arr_0"][0]
        lab.append(labtmp)
        X.append(np.array([d["arr_1"]+1.0]).T)
        Xw.append(np.array([d["arr_2"]+1.0]).T)
        G.append(np.array([d["arr_5"]+0.5]).T)
        info.append(d["arr_11"])
    lab=np.array(lab).astype(np.int32)
    X=np.array(X)
    Xw=np.array(Xw)
    G=np.array(G)
    info=np.array(info)
    
    return lab,X,Xw,G,info


def makearrB(flist,default_pick=0):
    #,[lab],lcs,lcsw,asinds,asindsw,infogap,infogapw,starinfo
    lab=[]
    X=[]
    Xw=[]
    info=[]
    for fn in flist:
        d=np.load(fn,allow_pickle=True)
        if d["arr_0"][0]==-1:
            labtmp=default_pick
        else:
            labtmp=d["arr_0"][0]
        lab.append(labtmp)
        X.append(np.array([d["arr_1"]+1.0,d["arr_3"]+1.0,d["arr_5"]+0.5,d["arr_7"]+1.0,d["arr_9"]+1.0]).T)
        Xw.append(np.array([d["arr_2"]+1.0,d["arr_4"]+1.0,d["arr_6"]+0.5,d["arr_8"]+1.0,d["arr_10"]+1.0]).T)                
        info.append(d["arr_11"])
    lab=np.array(lab).astype(np.int32)
    X=np.array(X)
    Xw=np.array(Xw)
    info=np.array(info)
    
    return lab,X,Xw,info


def makearrBa(flist,default_pick=0):
    #,[lab],lcs,lcsw,asinds,asindsw,infogap,infogapw,starinfo
    lab=[]
    X=[]
    Xw=[]
    info=[]
    for fn in flist:
        d=np.load(fn,allow_pickle=True)
        if d["arr_0"][0]==-1:
            labtmp=default_pick
        else:
            labtmp=d["arr_0"][0]
        lab.append(labtmp)
        X.append(np.array([d["arr_1"]+1.0,d["arr_3"]+1.0,d["arr_5"]+0.5]).T)
        Xw.append(np.array([d["arr_2"]+1.0,d["arr_4"]+1.0,d["arr_6"]+0.5]).T)                
        info.append(d["arr_11"])
    lab=np.array(lab).astype(np.int32)
    X=np.array(X)
    Xw=np.array(Xw)
    info=np.array(info)
    
    return lab,X,Xw,info

def makearrBL2(flist,default_pick=0):
    #LC and GAP
    #,[lab],lcs,lcsw,asinds,asindsw,infogap,infogapw,starinfo
    lab=[]
    X=[]
    Xw=[]
    info=[]
    for fn in flist:
        d=np.load(fn,allow_pickle=True)
        if d["arr_0"][0]==-1:
            labtmp=default_pick
        else:
            labtmp=d["arr_0"][0]
        lab.append(labtmp)
        X.append(np.array([d["arr_1"]+1.0,d["arr_5"]+0.5]).T)
        Xw.append(np.array([d["arr_2"]+1.0,d["arr_6"]+0.5]).T)                
        info.append(d["arr_11"])
    lab=np.array(lab).astype(np.int32)
    X=np.array(X)
    Xw=np.array(Xw)
    info=np.array(info)
    
    return lab,X,Xw,info


def makearrBL(flist,default_pick=0):
    #LC only
    #,[lab],lcs,lcsw,asinds,asindsw,infogap,infogapw,starinfo
    lab=[]
    X=[]
    Xw=[]
    info=[]
    for fn in flist:
        d=np.load(fn,allow_pickle=True)
        if d["arr_0"][0]==-1:
            labtmp=default_pick
        else:
            labtmp=d["arr_0"][0]
        lab.append(labtmp)
        X.append(np.array([d["arr_1"]+1.0]).T)
        Xw.append(np.array([d["arr_2"]+1.0]).T)                
        info.append(d["arr_11"])
    lab=np.array(lab).astype(np.int32)
    X=np.array(X)
    Xw=np.array(Xw)
    info=np.array(info)
    
    return lab,X,Xw,info


def makearr(flist,default_pick=0):
    #,[lab],lcs,lcsw,asinds,asindsw,infogap,infogapw,starinfo
    lab=[]
    X=[]
    Xw=[]
    info=[]
    for fn in flist:
        d=np.load(fn,allow_pickle=True)
        if d["arr_0"][0]==-1:
            labtmp=default_pick
        else:
            labtmp=d["arr_0"][0]
        lab.append(labtmp)
        X.append(np.array([d["arr_1"]+1.0,d["arr_3"]+1.0,d["arr_5"]+0.5]).T)
        Xw.append(np.array([d["arr_2"]+1.0,d["arr_4"]+1.0,d["arr_6"]+0.5]).T)                
        info.append(d["arr_7"])
    lab=np.array(lab).astype(np.int32)
    X=np.array(X)
    Xw=np.array(Xw)
    info=np.array(info)
    
    return lab,X,Xw,info

