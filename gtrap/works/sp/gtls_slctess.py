def injecttransit(lc,tu,nq,mstar,rstar):
    ############# INJECTION #################
    
    #f(P) propto P**-5/3
    xPmin=3.0
    xPmax=30.0    
    alpha=-5/3.
    a1=alpha+1.0
    Y = np.random.random(nq)
    Porb = ((xPmax**(a1) - xPmin**(a1) )*Y + xPmin**(a1))**(1/a1)

    #Radius
    xRpmin=0.2
    xRpmax=1.0    
    Y = np.random.random(nq)    
    Rp = Y*(xRpmax-xRpmin) + xRpmin
    #Rp= 1.0*np.ones(nq) ### DEBUG
    
    Mp = 1.0   
    Ms = mstar
    Rs = rstar

    a = (((Porb*u.day)**2 * G * (Ms*M_sun + Mp*M_jup) / (4*np.pi**2))**(1./3)).to(R_sun).value/Rs     
    b=np.random.random()
    ideg=np.arccos(b/a)/np.pi*180.0
    
    tux=np.copy(tu)
    tux[tu<0.0]=None
    tmin=np.nanmin(tux,axis=0)
    tmax=np.nanmax(tux,axis=0)

    
    t0 = np.random.random(nq)*(tmax-tmin)+tmin
    w = 0.0
    e = 0.0
    u1 = 0.1
    u2 = 0.3

    for i in range(0,nq):
        mask=(tu[:,i]>0.0)&(tu[:,i]==tu[:,i])
        ilc, b = gm.gentransit(tu[mask,i].astype(np.float64),t0[i],Porb[i],Rp[i],Mp,Rs[i],Ms[i],ideg[i],w,e,u1,u2)
        lc[mask,i] = lc[mask,i]*(2.0-ilc)

        fac=0.01
        ampS=(Rp[i]/Rs[i])**2*fac*np.nanmean(lc[mask,i])*np.random.random(1)
        ampC=0.5*ampS*np.random.random(1)
        #print(Porb[i],t0[i],ampS)
        ilsin=gm.gensin(tu[mask,i].astype(np.float64),Porb[i],t0[i],ampS)
        ilcos=gm.gendcos(tu[mask,i].astype(np.float64),Porb[i],t0[i],ampC)    
        lc[mask,i] = lc[mask,i] + ilsin + ilcos
    return lc, Porb, t0


if __name__ == "__main__":
    import astropy.units as u
    from astropy.constants import G, R_sun, M_sun, R_jup, M_jup

    import time
    import matplotlib.pyplot as plt
    import math
    import numpy as np
    import pycuda.autoinit
    import pycuda.driver as cuda
    import pycuda.compiler
    from pycuda.compiler import SourceModule
    import sys
    import gtrap.gtls as gtls
    import gtrap.gfilter as gfilter
    import gtrap.getstat as getstat
    import gtrap.read_keplerlc as kep
    import h5py
    import argparse
    import sqlite3
    import os
    import gtrap.tls as tls
    import gtrap.detect_peaks as dp
    import sys
    from time import sleep
    import pandas as pd
    import gtrap.genmock as gm
    import gtrap.picktrap as pt
#    import gtrap.read_tesslc as tes
    import gtrap.read_tesstic as tesstic
    import makefig
    
    start = time.time()

    parser = argparse.ArgumentParser(description='GPU Mock/Pick TESS TLS')
    parser.add_argument('-r', help='Randomly selected CTLv8/TIC/', action='store_true')
    parser.add_argument('-i', nargs=1, help='mid start (master ID)', type=int)
    parser.add_argument('-j', nargs=1, help='mid end (master ID)', type=int)

    parser.add_argument('-t', nargs='+', help='tic id', type=int)
    parser.add_argument('-scc', nargs='+', help='sector_camera_chip, ex) 1_1_1 ', type=str)
    parser.add_argument('-l', nargs=1, help='tic,scc list', type=str)

    # ex: python gtls_slctess.py -t 126786393 -scc 1_1_1 -n 2 -fig -q

    parser.add_argument('-m', nargs=1, default=[1],help='Mode: transit=0,lensing=1,absolute=2', type=int)
    parser.add_argument('-o', nargs=1, default=["output.txt"],help='output', type=str)
    parser.add_argument('-n', nargs=1, default=[1],help='Number of picking peaks', type=int)
    parser.add_argument('-fig', help='save figure', action='store_true')
    parser.add_argument('-c', help='Check detrended light curve', action='store_true')
    parser.add_argument('-smt', nargs=1, default=[15],help='smooth', type=int)
    parser.add_argument('-q', help='No injection', action='store_true')
    parser.add_argument('-p', help='picking mode', action='store_true')    
    parser.add_argument('-sd', nargs=1, help='output', type=str)
    parser.add_argument('-cb', nargs=1, help='counter of multibatch', type=int)
    
    ### SETTING
    ctldir="/home/kawahara/gtrap/data/ctl.list/"
    mpdin = 48 #1 d for peak search margin
    np.random.seed(1)            
    args = parser.parse_args()
    if args.sd:
        subd=args.sd[0]
    else:
        subd=""

    pickonly = args.p

    if args.i and args.j:
        midsx=args.i[0]
        midex=args.j[0]+1
        #    pickonly = False
        
        ###get filename from the list
        igname=os.path.join(ctldir, "igtrap.list")
        datc=pd.read_csv(igname,names=("tag","i","j"))
        
        taglist=datc["tag"]
        tagnum=datc["i"].values
        
        itag=np.searchsorted(tagnum,midsx+1)
        itage=np.searchsorted(tagnum,midex+1)
        print(itag,itage)
        
        if itag==itage:    
            tag=taglist[itag-1]
            print("tag",tag,midsx,midex)
            listname=os.path.join(ctldir,"ctl.list_"+tag+".npz")
            dat=np.load(listname)
            igtrap=dat["arr_0"]
            mids=np.searchsorted(igtrap,midsx)
            mide=np.searchsorted(igtrap,midex)
            filelist=dat["arr_1"][mids:mide]    
            rstar=dat["arr_2"][mids:mide] #stellar radius
            mstar=dat["arr_3"][mids:mide] #stellar mass
            
        else:
            tag=taglist[itag-1]        
            listname=os.path.join(ctldir,"ctl.list_"+tag+".npz")
            dat=np.load(listname)
            igtrap=dat["arr_0"]
            mids=np.searchsorted(igtrap,midsx)
            filelist=dat["arr_1"][mids:]    
            
            rstar=dat["arr_2"][mids:] #stellar radius
            mstar=dat["arr_3"][mids:] #stellar mass        
            
            tag=taglist[itage-1]
            listname=os.path.join(ctldir,"ctl.list_"+tag+".npz")
            dat=np.load(listname)
            igtrap=dat["arr_0"]
            mide=np.searchsorted(igtrap,midex)
            filelist=np.concatenate([filelist,dat["arr_1"][:mide]])
            rstar=np.concatenate([rstar,dat["arr_2"][:mide]]) #stellar radius
            mstar=np.concatenate([mstar,dat["arr_3"][:mide]]) #stellar mass        
    elif (args.t and args.scc) or args.l:
        from urllib.parse import urlparse
        import mysql.connector

        if args.l:
            tslist=pd.read_csv(args.l[0],delimiter=",")
            ticlist=tslist["TIC"]
            scclist=tslist["SCC"]
        else:
            ticlist=args.t
            scclist=args.scc
            
        url = urlparse('mysql://fisher:atlantic@133.11.229.168:3306/TESS')
        conn = mysql.connector.connect(
            host = url.hostname or '133.11.229.168',
            port = url.port or 3306,
            user = url.username or 'fisher',
            password = url.password or 'atlantic',
            database = url.path[1:],
        )
        cur = conn.cursor()

        rstar=[]
        mstar=[]
        filelist=[]
        for ii,tic in enumerate(ticlist):
            scc=scclist[ii]
            com='SELECT rad,mass FROM CTLchip'+scc+' where ID='+str(tic)
            sector = int(scc.split("_")[0])
            print(com,sector)
            if sector<11:
                filelist.append("/manta/pipeline/CTL2/tess_"+str(tic)+"_"+str(scc)+".h5")
            else:
                filelist.append("/stingray/pipeline/CTL2/tess_"+str(tic)+"_"+str(scc)+".h5")

            cur.execute(com)
            out=cur.fetchall()[0]
            out=np.array(out) #rad, mass
            rstar.append(out[0])
            mstar.append(out[1])
        rstar=np.array(rstar)
        mstar=np.array(mstar)

    else:
        sys.exit("Specify -i&-j or -t&-scc")

    nin=2000
    lc,tu,asind,lcicb1,lcicb2,n,ntrue,nq,inval,tu0,ticarr,sectorarr, cameraarr, CCDarr=tesstic.load_tesstic(filelist,nin,offt="t[0]",nby=1000)
    icnt=0
    idet=0 #detected number
    
    if args.q or pickonly:
        print("NO INJECTION")
    else:
        lc,Porb,t0=injecttransit(lc,tu,nq,mstar,rstar)
        
    ## for transit ##
    if args.m[0] == 0:
        lc = 2.0 - lc
        print("Transit Mode")
    elif args.m[0] == 1:
        print("Lensing Mode")
    elif args.m[0] == 2:
        print("Absolute Mode")
    else:
        sys.exit("No mode")

    #median filter width
    #medsmt_width = 50
    medsmt_width = 50
    nby=1000 ## of thread
    dev_imgout=gfilter.get_detrend(lc,nby=nby,nbx=1,r=medsmt_width,isw=0,osw=1) #detrend
    
    #set
    tu=np.array(tu,order="C").astype(np.float32)

    #determine deltat
    ttmp=tu[:,0]
    numl=np.array(list(range(0,len(ttmp))))
    mask=(ttmp>=0.0)
    deltat=(ttmp[mask][-1] - ttmp[mask][0])/(numl[mask][-1] - numl[mask][0])

    #tbls setting
    #the number of the window width= # of the threads should be 2^n !!!!
    nw=1024 ### ? 
    
    # Max and Min of Widths
    wmin = 0.20  #  day
    wmax = 1.0  # day
    dw=(wmax-wmin)/(nw-1)
    t0min=(2*wmin) #test
#    t0min=(0.*wmin) #test
    
    dt=1.0
    nt=len(tu[:,0])
    nl=50

    # the number of a scoop
    nsc=int(wmax/deltat+3.0)
    dev_tu = cuda.mem_alloc(tu.astype(np.float32).nbytes)
    cuda.memcpy_htod(dev_tu,tu.astype(np.float32))

    ntrue=ntrue.astype(np.int32)
    dev_ntrue = cuda.mem_alloc(ntrue.nbytes)
    cuda.memcpy_htod(dev_ntrue,ntrue)
    
    #output TLS s/n, w, l
    tlssn=np.zeros(nt*nq).astype(np.float32)
    dev_tlssn = cuda.mem_alloc(tlssn.nbytes)
    tlsw=np.zeros(nt*nq).astype(np.float32)
    dev_tlsw = cuda.mem_alloc(tlsw.nbytes)
    tlst0=np.zeros(nt*nq).astype(np.float32)
    dev_tlst0 = cuda.mem_alloc(tlst0.nbytes)
    tlsl=np.zeros(nt*nq).astype(np.float32)
    dev_tlsl = cuda.mem_alloc(tlsl.nbytes)
    tlshmax=np.zeros(nt*nq).astype(np.float32)
    dev_tlshmax = cuda.mem_alloc(tlshmax.nbytes)

    source_module=gtls.gtls_module()
    
    sharedsize=(2*nsc + nw + 2)*4 #byte
    if args.m[0] == 2:
        source_module=gtls.gtls_module("absolute")
    else:
        source_module=gtls.gtls_module()
    pkernel=source_module.get_function("gtls")

    pkernel(dev_tlssn,dev_tlsw,dev_tlst0,dev_tlsl,dev_tlshmax,\
            dev_imgout,dev_tu,\
            np.int32(nt),\
            np.int32(nl),np.int32(nsc),\
            np.float32(t0min),np.float32(dt),\
            np.float32(wmax),np.float32(dw),np.float32(deltat),\
            block=(int(nw),1,1), grid=(int(nt),int(nq)),shared=sharedsize)
        
    cuda.memcpy_dtoh(tlssn, dev_tlssn)
    cuda.memcpy_dtoh(tlsw, dev_tlsw)
    cuda.memcpy_dtoh(tlst0, dev_tlst0)
    cuda.memcpy_dtoh(tlsl, dev_tlsl)
    cuda.memcpy_dtoh(tlshmax, dev_tlshmax)

    #========================================--

    ########################
    PickPeaks=args.n[0]
    for iq,tic in enumerate(ticarr):
        print(iq,"/",len(ticarr))
        if True:
#        try:
            ticname=str(tic)
            detection=0
            lab=0
            fac=1.0
            ffac = 8.0 #region#
            
            mask = (tlssn[iq::nq]>0.0)        
            std=np.std(tlssn[iq::nq][mask])
            median=np.median(tlssn[iq::nq])
            #### PEAK STATISTICS ####
            peak = dp.detect_peaks(tlssn[iq::nq],mpd=mpdin)
            peak = peak[np.argsort(tlssn[iq::nq][peak])[::-1]]                
            PickPeaks=min(PickPeaks,len(tlssn[iq::nq][peak]))
            maxsn=tlssn[iq::nq][peak][0:PickPeaks]
            far=(maxsn-median)/std
            minlen =  10000.0 #minimum length for time series
            lent =len(tlssn[iq::nq][tlssn[iq::nq]>0.0])
            detsw=0
            for ipick in range(0,PickPeaks):
                i = peak[ipick]
                im=np.max([0,int(i-nsc*ffac)])
                ix=np.min([nt,int(i+nsc*ffac)])
                imn=np.max([0,int(i-nsc/2)])
                ixn=np.min([nt,int(i+nsc/2)])
                            
                if args.m[0] == 0:    
                    llc=2.0 - lc[im:ix,iq]
                    llcn=2.0 - lc[imn:ixn,iq]
                    
                elif args.m[0] == 1 or args.m[0] == 2:
                    llc=lc[im:ix,iq]
                    llcn=lc[imn:ixn,iq]
            
                ttc=tu[im:ix,iq]
                ttcn=tu[imn:ixn,iq]#narrow region
                
                #PEAK VALUE
                T0=tlst0[iq::nq][peak[ipick]]+tu0[iq]
                W=tlsw[iq::nq][peak[ipick]]
                L=tlsl[iq::nq][peak[ipick]]
                H=tlshmax[iq::nq][peak[ipick]]
                
                #################
                #print("GPU ROUGH: T0,W,L,H")
                #print(T0,W,L,H)
                
                if args.q or pickonly:
                    dTpre=0.0
                else:
                    dTpre=np.abs((np.mod(T0,Porb[iq]) - np.mod(t0[iq]+tu0[iq],Porb[iq]))/(W/2))
                #print("DIFF/dur=",dTpre)

#                if True:
                if (dTpre < 0.1 and detection == 0) or args.q or pickonly:
                    #print("(*_*)/ DETECTED at n=",ipick+1," at ",peak[ipick])
                    detection = ipick+1
                    if args.q:
                        lab=0
                    elif pickonly:
                        lab=-1
                    else:
                        lab=1
                    
                    
                    T0tilde=tlst0[iq::nq][peak[ipick]]
                    ## REINVERSE
                    if args.m[0] == 0:
                        lc = 2.0 - lc

                    sector=sectorarr[iq]
                    camera=cameraarr[iq]
                    CCD=CCDarr[iq]
                    #scctag=str(sector)+"_"+str(camera)+"_"+str(CCD)

                    if pickonly:
                        tag="TIC"+str(ticname)+"s"+str(lab)
                        savedd="/home/kawahara/gtrap/examples/picklcslc/"+subd

                    else:
                        if args.cb:
                            counter=str(args.cb[0])+"_"+str(icnt)
                        else:
                            counter=str(icnt)
                        tag=counter+"_TIC"+str(tic)+"_"+str(ipick)+"TF"+str(lab)
                        savedd="/home/kawahara/gtrap/examples/mocklcslc/"+subd

                    savpng=os.path.join(savedd,"png")                       
                    savnpz=os.path.join(savedd,"npz")                       

                    os.makedirs(savnpz, exist_ok=True)
                    os.makedirs(savpng, exist_ok=True)

                        
                    ticname=str(tic)+"_"+str(sector)+"_"+str(camera)+"_"+str(CCD)

                    lcicb1s, c1tus, c1infogap, c1prec=pt.pick_Wnormalized_cleaned_lc_direct(lcicb1[:,iq],tu[:,iq],T0tilde,W,alpha=1,nx=51,check=args.fig,tag=tag+"_c1local",savedir=savpng,T0lab=T0)                        
                    lcicb2s, c2tus, c2infogap, c2prec=pt.pick_Wnormalized_cleaned_lc_direct(lcicb2[:,iq],tu[:,iq],T0tilde,W,alpha=1,nx=51,check=args.fig,tag=tag+"_c2local",savedir=savpng,T0lab=T0)                        

                    asinds, atus, ainfogap, aprec=pt.pick_Wnormalized_cleaned_lc_direct(asind[:,iq],tu[:,iq],T0tilde,W,alpha=1,nx=51,check=args.fig,tag=tag+"_alocal",savedir=savpng,T0lab=T0)                        
                    lcs, tus, infogap, prec=pt.pick_Wnormalized_cleaned_lc_direct(lc[:,iq],tu[:,iq],T0tilde,W,alpha=1,nx=51,check=args.fig,tag=tag+"_local",savedir=savpng,T0lab=T0)

                    lcicb1sw, c1tusw, c1infogapw, c1precw=pt.pick_Wnormalized_cleaned_lc_direct(lcicb1[:,iq],tu[:,iq],T0tilde,W,alpha=5,nx=251,check=args.fig,tag=tag+"_c1wide",savedir=savpng,T0lab=T0)                        
                    lcicb2sw, c2tusw, c2infogapw, c2precw=pt.pick_Wnormalized_cleaned_lc_direct(lcicb2[:,iq],tu[:,iq],T0tilde,W,alpha=5,nx=251,check=args.fig,tag=tag+"_c2wide",savedir=savpng,T0lab=T0)                        
                    
                    asindsw, atusw, ainfogapw, aprecw=pt.pick_Wnormalized_cleaned_lc_direct(asind[:,iq],tu[:,iq],T0tilde,W,alpha=5,nx=251,check=args.fig,tag=tag+"_awide",savedir=savpng,T0lab=T0)
                    lcsw, tusw, infogapw, precw=pt.pick_Wnormalized_cleaned_lc_direct(lc[:,iq],tu[:,iq],T0tilde,W,alpha=5,nx=251,check=args.fig,tag=tag+"_wide",savedir=savpng,T0lab=T0)
                    
                    starinfo=[mstar,rstar,tic,sector,camera,CCD,T0,W,L,H]
                    if pickonly:
                        np.savez(os.path.join(savnpz,"pick"+str(ticname)+"_"+str(ipick)),[lab],lcs,lcsw,asinds,asindsw,infogap,infogapw,lcicb1s,lcicb1sw,lcicb2s,lcicb2sw,starinfo)
                    else:
                        np.savez(os.path.join(savnpz,counter+"_mock"+str(ticname)+"_"+str(ipick)+"TF"+str(lab)),[lab],lcs,lcsw,asinds,asindsw,infogap,infogapw,lcicb1s,lcicb1sw,lcicb2s,lcicb2sw,starinfo)
                    icnt=icnt+1
                    if detsw == 0:
                        idet=idet+1
                        detsw=0
#        except:
#            print(iq,"/",len(ticarr),"Some Error in cleanning/")

    print("Detected:",idet,"/",len(ticarr))

    elapsed_time = time.time() - start
    print (("2 :{0}".format(elapsed_time)) + "[sec]")
#    print (elapsed_time/(midex-midsx), "[sec/N]")
