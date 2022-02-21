import numpy as np
def gen_trapzoid(t,H,W,L,t0):
    tn=t-t0
    x=np.zeros(len(tn))
    maskAx=(tn<=-W/4-L/2)
    maskB=(tn>-W/4-L/2)*(tn<=-W/4+L/2)
    maskC=(tn<W/4-L/2)*(tn>=-W/4+L/2)
    maskD=(tn<W/4+L/2)*(tn>=W/4-L/2)
    maskEx=(tn>=W/4+L/2)
    x[maskAx]=-H/2
    x[maskEx]=-H/2   
    x[maskC]=H/2      
    x[maskB]=H/L*(tn[maskB]+W/4)
    x[maskD]=-H/L*(tn[maskD]-W/4)
    return x

def get_Hmax(t,x,W,L,t0):
    tn=t-t0
    maskA=(tn>=-W/2)*(tn<-W/4-L/2)
    maskB=(tn>=-W/4-L/2)*(tn<-W/4+L/2)
    maskC=(tn<W/4-L/2)*(tn>=-W/4+L/2)
    maskD=(tn<W/4+L/2)*(tn>=W/4-L/2)
    maskE=(tn<W/2)*(tn>=W/4+L/2)
    nA=len(t[maskA])
    nB=len(t[maskB])
    nC=len(t[maskC])
    nD=len(t[maskD])
    nE=len(t[maskE])
    #print(nA,nB,nC,nD,nE)
    xAEC=np.nansum(x[maskA])+np.nansum(x[maskE])+np.nansum(x[maskC])
    xBD=np.nansum(x[maskB])+np.nansum(x[maskD])
    xtB=np.nansum(x[maskB]*tn[maskB])
    xtD=np.nansum(x[maskD]*tn[maskD])
    tB=np.nansum(tn[maskB])
    tD=np.nansum(tn[maskD])
    t2BD=np.nansum(tn[maskB]*tn[maskB])+np.nansum(tn[maskD]*tn[maskD])
    #print(xAEC+xBD)
    #denA=4*L*L*(nA-nC+nE)+W*W*(nB+nD)+8*W*(tB-tD)+16*t2BD
    denA=4*L*L*(nA-nC+nE)+W*W*(nB+nD)+8*W*(tB-tD)+16*t2BD
    numB=8*L*L*xAEC-4*W*L*xBD+16*L*(xtD-xtB)
    Hmax=-numB/denA
    return Hmax

def get_chi2Hmax(t,xx,W,L,t0,sigma=1.0,divx=True):
    tn=t-t0
    maskA=(tn>=-W/2)*(tn<-W/4-L/2)
    maskB=(tn>=-W/4-L/2)*(tn<-W/4+L/2)
    maskC=(tn<W/4-L/2)*(tn>=-W/4+L/2)
    maskD=(tn<W/4+L/2)*(tn>=W/4-L/2)
    maskE=(tn<W/2)*(tn>=W/4+L/2)
    nA=len(t[maskA])
    nB=len(t[maskB])
    nC=len(t[maskC])
    nD=len(t[maskD])
    nE=len(t[maskE])
    n=(nA+nB+nC+nD+nE)
    if divx:
        xsum=np.nanmean(xx[(tn>=-W/2)*(tn<=W/2)])
        x=np.copy(xx)-xsum
    else:
        xsum=0.0
        x=xx
        
    xAEC=np.nansum(x[maskA])+np.nansum(x[maskE])+np.nansum(x[maskC])
    xBD=np.nansum(x[maskB])+np.nansum(x[maskD])
    xtB=np.nansum(x[maskB]*tn[maskB])
    xtD=np.nansum(x[maskD]*tn[maskD])
    tB=np.nansum(tn[maskB])
    tD=np.nansum(tn[maskD])
    t2BD=np.nansum(tn[maskB]*tn[maskB])+np.nansum(tn[maskD]*tn[maskD])
    denA=4*L*L*(nA-nC+nE)+W*W*(nB+nD)+8*W*(tB-tD)+16*t2BD
    numB=8*L*L*xAEC-4*W*L*xBD+16*L*(xtD-xtB)
    Hmax=-numB/denA
    Hmaxh=Hmax/2.0
    res=np.nansum((x[maskA]+Hmaxh)**2)+np.nansum((x[maskC]-Hmaxh)**2)+np.nansum((x[maskE]+Hmaxh)**2)
    res=res+np.nansum((x[maskB] - Hmax/L*(tn[maskB]+W/4.0))**2)
    res=res+np.nansum((x[maskD] + Hmax/L*(tn[maskD]-W/4.0))**2)
    res=res/(sigma*sigma)
    return Hmax, res, n, xsum


def tlsfit(t,x,Wl,Ll,tl):
    arr=[]
    arrt0=[]
    arrW=[]
    arrL=[]
    sigma=1.0
    for iW in Wl:
        for iL in Ll:
            for it0 in tl:
                d=get_chi2Hmax(t,x,iW,iL,it0,sigma)
                arr.append(d)
                arrt0.append(it0)
                arrW.append(iW)
                arrL.append(iL)
                
    arr=np.array(arr)
    arrt0=np.array(arrt0)
    arrL=np.array(arrL)
    sn=-arr[:,0]/np.sqrt(arr[:,1]/(arr[:,2]-3))
    #Hmax, res, n, xsum
    imax=np.nanargmax(sn)
    print(imax)
    H=arr[imax,0]
    W=arrW[imax]
    L=arrL[imax]
    T0=arrt0[imax]
    offset=arr[imax,3]
    xpre=gen_trapzoid(t,H,W,L,T0)+offset
    return xpre,H,W,L,T0,offset

    
if __name__ == "__main__":

    t=np.linspace(0,100,132)
    t0=50
    H=-3.0
    W=30.0
    L=5
    x=gen_trapzoid(t,H,W,L,t0)
    print("input H=",H,"output H=",get_Hmax(t,x,W,L,t0))
    print("input H=",H,"output H=",get_chi2Hmax(t,x,W,L,t0))
