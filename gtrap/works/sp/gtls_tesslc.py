import re
def numericalSort(value):
    numbers = re.compile(r'(\d+)')
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts

if __name__ == "__main__":
    import time
    import matplotlib.pyplot as plt
    import math
    import numpy as np
    import pycuda.autoinit
    import pycuda.driver as cuda
    import pycuda.compiler
    from pycuda.compiler import SourceModule
    import sys
    import gtls_simple
    import gtls_absolute
    import gfilter
    import getstat
    import read_tesslc as tes
    import glob
    import argparse
    import tls
    import detect_peaks as dp
    import os
    from time import sleep
    import gpucheck
    import pandas as pd
    start = time.time()

    parser = argparse.ArgumentParser(description='GPU Kepler TLS')
    parser.add_argument('-i', nargs=1, help='mid (master ID)', type=int)
    parser.add_argument('-d', nargs=1, default=["data/master.list"],help='master list', type=str)
    
    parser.add_argument('-t', nargs='+', help='tic id', type=int)
    parser.add_argument('-q', nargs=1, default=["BLS_TEST"],help='SQL table name', type=str)
    parser.add_argument('-b', nargs=1, default=[0],help='batch num', type=int)
    parser.add_argument('-f', nargs=1, default=["fits"],help='filetype: fits (BKJD), hdf5 (relative Day)', type=str)
    parser.add_argument('-m', nargs=1, default=[0],help='Mode: transit=0,lensing=1,absolute=2', type=int)
    parser.add_argument('-o', nargs=1, default=["output.txt"],help='output', type=str)
    parser.add_argument('-n', nargs=1, default=[1],help='The target of the number of the transits: STE=1, DTE=2 for the max SN. ', type=int)
    #    parser.add_argument('-p', nargs=1, default=[400],help='Minimum interval for DTE (d)', type=float)
    parser.add_argument('-fig', help='save figure', action='store_true')
    parser.add_argument('-c', help='Check detrended light curve', action='store_true')

    parser.add_argument('-gpu', nargs=3, default=[85.0,51.0,30.0],help='GPU control, limit temperature [K], settle temperature, sleep time [sec] for cool down', type=float)
    parser.add_argument('-smt', nargs=1, default=[15],help='smooth', type=int)

    ### SETTING
    mpdin = 48 #1d for peak search margin
    n=1 #only 1 target can be analyzed
            
    # #
    args = parser.parse_args()

    #### GPU HEALTH CHECK #######
    maxt=args.gpu[0]
    sett=args.gpu[1]
    sleeptime=args.gpu[2]
    maxgput=gpucheck.getmaxgput_gput()
    print(maxgput,"[C] for init")
    if maxgput > maxt:
        print("Fever. Waiting for cool down of the fevered GPU. ")
        fever = True
        while fever :
            sleep(sleeptime)
            maxgput=gpucheck.getmaxgput_gput()
            print(maxgput,"[C]")
            if maxgput < sett:
                fever = False
    #### SLEEP TIME #######


    if args.i:
        masterlist=args.d[0]
        print("MASTER LIST=",masterlist)
        mdat=pd.read_csv(masterlist,delimiter=",")
        mid=args.i[0]
        print("MID=",mid)
        mask=mdat["MID"]==mid
        print(mdat["ID"][mask].values[0])
        ticint=mdat["ID"][mask].values[0]
    elif args.t:
        ticint=args.t[0]
    
    lc=[]
    tu=[]


    ###generate filelist
    icdir="/pike/tess/data/lc"
    print(icdir+'/*'+str(ticint)+'*.fits')
    ff=sorted(glob.glob(icdir+'/*'+str(ticint)+'*.fits'), key=numericalSort)
    filelist=[]
    for filef in ff:
        filelist.append(os.path.join(icdir,filef))
    print(filelist)
    print("fits mode, BKJD.")
    lc,tu,n,ntrue,nq,inval,bjdoffset,t0arr, t, det, info=tes.load_tesslc(filelist)
    tu0=t0arr

    print("number of the KICs, nq=",nq)
    print("offset=",tu0)
    ##OFFSET


    ## for transit ##
    if args.m[0] == 0:
        lc = 2.0 - lc
        figtag="dip"
        print("Transit Mode")
    elif args.m[0] == 1:
        figtag="pulse"
        print("Lensing Mode")
    elif args.m[0] == 2:
        figtag="abs"
        print("Absolute Mode")
    else:
        sys.exit("No mode")
    ###############
    
    elapsed_time = time.time() - start
    print (("2 :{0}".format(elapsed_time)) + "[sec]")

    ##detrend (directly use)
    nby=1000 ## of thread
    dev_imgout=gfilter.get_detrend(lc,nby=nby,nbx=1,r=512,isw=0,osw=1) #detrend

    if args.c:
        imgout= np.zeros(np.shape(lc.flatten())).astype(np.float32)
        cuda.memcpy_dtoh(imgout, dev_imgout)
        plt.plot(tu,imgout,".")
        plt.show()
    
    #set
    tu=np.array(tu,order="C").astype(np.float32)

    #determine deltat
    ttmp=tu[:,0]
    numl=np.array(list(range(0,len(ttmp))))
    mask=(ttmp>=0.0)
    deltat=(ttmp[mask][-1] - ttmp[mask][0])/(numl[mask][-1] - numl[mask][0])
    #tbls setting
    #the number of the window width= # of the threads should be 2^n !!!!
    nw=128 
    
    # Max and Min of Widths
    wmin = 0.1  #  day
    wmax = 1.5  # day
    dw=(wmax-wmin)/(nw-1)
    t0min=(2*wmin) #test
    t0max=n-2*wmin #test

    dt=1.0
    nt=len(tu[:,0])
    #L
    nl=10

    # the number of a scoop
    nsc=int(wmax/deltat+3.0)

    print(nl,nt,nsc,nw)
    print("# of threads (W) = ",nw)
    print("# of for-loop (L) = ",nl)
    print("scoop number = ",nsc)
    

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

    source_module=gtls_simple.gtls_module()
    
    ##compute kma,kmi,kkmi
    sharedsize=(2*nsc + nw + 2)*4 #byte
    print("sharedsize=",sharedsize)
    #gtls
    start = time.time()
    if args.m[0] == 2:
        source_module=gtls_absolute.gtls_module()
    else:
        source_module=gtls_simple.gtls_module()
    pkernel=source_module.get_function("gtls")
    print(nw,nt,nq,nsc,dt,t0min,wmax,dw,deltat,sharedsize)
    
    pkernel(dev_tlssn,dev_tlsw,dev_tlst0,dev_tlsl,dev_tlshmax,\
            #dev_debugt,dev_debuglc,dev_debugpar,\
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
#    dat=np.load("data/cdpp15.npz")
#    kicintccdp=np.array(dat["arr_0"][:,0]).astype(np.int)
#    ccdp15=np.array(dat["arr_0"][:,1])

    ########################
    PeakMaxindex=args.n[0]-1
    iq=0


    fac=1.0
    ffac = 8.0 #region#

    
    mask = (tlssn[iq::nq]>0.0)        
    std=np.std(tlssn[iq::nq][mask])
    median=np.median(tlssn[iq::nq])
    #### PEAK STATISTICS ####
    peak = dp.detect_peaks(tlssn[iq::nq],mpd=mpdin)
    peak = peak[np.argsort(tlssn[iq::nq][peak])[::-1]]        
    ntop = 4 # the nuber of the top peaks

    #===========CRIT 1==========================       
    if len(peak) > ntop:
        peak=peak[0:ntop]
        peakval = tlssn[iq::nq][peak]
        print(peakval)
        ## dntop-1-th peak/std
        diff=(peakval[-1]-median)/std
    else:
        ntop=len(peak)            
        diff = 0.0
    #===========================================
    
    maxsn=tlssn[iq::nq][peak][PeakMaxindex]
    Pinterval=np.abs(tlst0[iq::nq][peak][1]-tlst0[iq::nq][peak][0])
    far=(maxsn-median)/std

    minlen =  10000.0 #minimum length for time series
    lent =len(tlssn[iq::nq][tlssn[iq::nq]>0.0])
    
    #        if maxsn > 1.8:
    #            xxcrit=10.5*np.log10(maxsn-1.8)**0.6-3.5
    #            #xxcrit=10.5*np.log10(maxsn-1.8)**0.6-1.5
    #            xxcrit=10.5*np.log10(maxsn-1.8)**0.6-0.9 #DTE
    #        else:
    #            xxcrit=0.0
    
    if True:
        #        if (diff < xxcrit and diff < 8.0 and float(lent) > minlen and Pinterval > 300.0) or args.n[0]==1:
        i = np.argmax(tlssn[iq::nq])
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
        T0p=tlst0[iq::nq][peak][0]+tu0[iq]
        Wp=tlsw[iq::nq][peak][0]
        Lp=tlsl[iq::nq][peak][0]
        Hp=tlshmax[iq::nq][peak][0]

        #################
        print("GPU ROUGH: T0,W,L,H")
        print(T0p,Wp,Lp,Hp)
        ### RE (PRECISE) FIT
        NT0=10
        dT0=0.1
        tl=np.linspace(T0p-dT0,T0p+dT0,NT0)
        NW=10
        dW=Wp/100
        Wl=np.linspace(Wp-dW,Wp+dW,NW)
        NL=20
        Lpd=0.0
        Lpu=Wp/2.0
        Ll=np.linspace(Lpd,Lpu,NL)
        if args.m[0] == 0:    
            detpre,H,W,L,T0,offsetlc=tls.tlsfit(t,det,Wl,Ll,tl)
        elif args.m[0] == 1:    
            detpre,H,W,L,T0,offsetlc=tls.tlsfit(t,2.0-det,Wl,Ll,tl)

        H=-H
        print("PRECISE TRAFIT: T0,W,L,H")
        print(T0,W,L,H)
        
        if args.o:
            ttag=args.o[0].replace(".txt","")
        else:
            ttag="_"

        ofile=args.o[0]
        f = open(ofile, 'a')
        f.write(str(ticint)+","+str(maxsn)+","+str(far)+","+str(diff)+","+str(lent)+","+str(Pinterval)+","+str(H)+","+str(tlst0[iq::nq][i]+tu0[iq])+","+str(L)+","+str(W)+"\n")
        f.close()
            
            
        ff = open("slcinfo"+ttag+".txt", 'a')
        ff.write(str(ticint)+","+str(H)+","+str(tlst0[iq::nq][i]+tu0[iq])+","+str(L)+","+str(W)+",\n")
        ff.close()
            
        ###############################################
        
        if args.fig:
            teff=info[0]["TEFF"]
            rads=info[0]["RADIUS"]
            fig=plt.figure(figsize=(10,7.5))
            ax=fig.add_subplot(3,1,1)

            num=args.smt[0]
            b=np.ones(num)/num
            det2=np.convolve(det, b, mode='same')
            plt.ylim(np.nanmin(det),np.nanmax(det))
            ax.plot(t,det,".",color="gray",alpha=0.1)            
            ax.plot(t,det2,color="black",alpha=0.7)

            plt.tick_params(labelsize=18)
            #plt.plot(tlst0[iq::nq]+tu0[iq],tlssn[iq::nq],color="gray")
            #plt.plot(tlst0[iq::nq][peak]+tu0[iq],tlssn[iq::nq][peak],"v",color="red",alpha=0.3)
            plt.axvline(tlst0[iq::nq][i]+tu0[iq],color="red",alpha=0.3)
            #plt.axhline(median,color="green",alpha=0.3)
            plt.ylabel("PDCSAP",fontsize=18)            
            #            plt.title("TIC"+str(ticint)+" SN="+str(round(maxsn,1))+" far="+str(round(far,1))+" dp="+str(round(diff,1)))
            plt.title("TIC"+str(ticint)+" "+str(int(teff))+"K "+str(round(rads,2))+"Rsol",fontsize=18)
            ax=fig.add_subplot(3,1,2)
            plt.tick_params(labelsize=18)
            plt.axvline(tlst0[iq::nq][i]+tu0[iq],color="red",alpha=0.3)
            
            plt.xlim(tlst0[iq::nq][i]-tlsw[iq::nq][i]*ffac+tu0[iq],tlst0[iq::nq][i]+tlsw[iq::nq][i]*ffac+tu0[iq])
            ax.plot(tu[im:ix,iq]+tu0[iq],llc,".",color="gray",alpha=0.3)
            
            ##==================##
            
            mask=(ttc>0.0)
            dip=np.abs((np.max(llc[mask])-np.min(llc[mask]))/3.0)
            maskn=(ttcn>0.0)
            
            plt.ylim(np.min(llcn[maskn])-dip,np.max(llcn[maskn])+dip)
            plt.ylabel("PDCSAP",fontsize=18)
            
            ax=fig.add_subplot(3,1,3)
            plt.tick_params(labelsize=18)
            
            pre=tls.gen_trapzoid(ttc[mask]+tu0[iq],H,W,L,T0)
            if args.m[0] == 0:    
                pre = 1.0 - pre
            elif args.m[0] == 1:    
                pre = 1.0 + pre
            ax.plot(ttc[mask]+tu0[iq],pre-(1.0-offsetlc),color="green")
            ax.plot(ttc[mask]+tu0[iq],pre-(1.0-offsetlc),".",color="green",alpha=0.3)
            
            plt.xlim(tlst0[iq::nq][i]-tlsw[iq::nq][i]*ffac+tu0[iq],tlst0[iq::nq][i]+tlsw[iq::nq][i]*ffac+tu0[iq])
            plt.ylim(np.min(llcn[maskn])-dip,np.max(llcn[maskn])+dip)
            plt.ylabel("best matched",fontsize=18)
            if tu0[iq]==0.0:
                plt.xlabel("Day",fontsize=18)
            else:
                plt.xlabel("BKJD",fontsize=18)

            plt.savefig("TIC"+str(ticint)+figtag+".png")
            print("t0=",tlst0[iq::nq][i]+tu0[iq])
            print("height=",H)
            print("L=",L)
            


#            plt.savefig("TIC"+str(ticint)+".pdf", bbox_inches="tight", pad_inches=0.0)
#            plt.show()

