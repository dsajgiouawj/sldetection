#/usr/bin/python
import matplotlib.pyplot as plt
import math
import numpy as np
import argparse
from astropy.io import fits
import gtrap.read_keplerlc as kep
import pandas as pd
from scipy import signal 
import os
from scipy import interpolate
import gtrap.picktrap as pt

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='pick up trapezoid candidate for classification (generating cleaned light curves around dips)')
    parser.add_argument('-f', nargs=1, help='info file', type=str)
    parser.add_argument('-n', nargs="+", help='csv file number. needed to use -f. minus value = use all the numbers in f.', type=int)
    parser.add_argument('-nn', nargs=2, help='use -f from n[0] to n[1]', type=int)
    parser.add_argument('-k', nargs=1, default=[1717722], help='kic', type=int)
    parser.add_argument('-t', nargs=1, default=[1439.2],help='time (T0) [BKJD]', type=float)
    parser.add_argument('-w', nargs=1, default=[2.0],help='width [BKJD]', type=float)

    lctag="data"
    mydir="/sharksuck/kic/";
    args = parser.parse_args()
    if args.n:
        nidarr=args.n
        if nidarr[0] < 0:
            dat=pd.read_csv(args.f[0],comment="#")            
            try:
                nidarr=dat["number"].values
            except:
                nidarr=np.asarray(range(0,len(dat)))
    else:
        nidarr=[-1]

    lcsall=[]
    lcsallw=[]
        
    if args.nn:
        dat=pd.read_csv(args.f[0],comment="#")
        try:
            nidarr=dat["number"].values[args.nn[0]:args.nn[1]]
        except:
            nidarr=np.asarray(range(0,len(dat)))[args.nn[0]:args.nn[1]]


    print(nidarr)
    for ii,nid in enumerate(nidarr):
        
        if args.f:        
            dat=pd.read_csv(args.f[0],comment="#")
            if args.n or args.nn:
                try:
                    mask=dat["number"]==nid
                    kicint=dat["KIC"][mask].values[0]
                except:
                    kicint=dat["KIC"].values[ii]
                    mask=dat["KIC"]==kicint
                print("KIC",kicint)
            else:
                kicint=args.k[0]
                mask=dat["KIC"]==kicint
                
            T0=dat["T0BKJD"][mask].values[0]
            W=dat["W"][mask].values[0]
        else:
            kicint=args.k[0]
            T0=args.t[0]
            W=args.w[0]

        ######################################################################
            
        print("*********",kicint,"=",nid,"*********")
        kicdir=pt.getkicdir(kicint,mydir+"data/")+"/"

        lcs, tus, prec=pt.pick_cleaned_lc(kicdir,T0,wid=128,check=True,tag="KIC"+str(kicint),savedir=args.f[0].replace(".txt",""))
        
        if len(lcs[lcs==lcs])>0 and prec:
            lcsall.append(lcs)
            print("1. CLEANED LC IS APPENDED")

        lcsw, tusw, precw=pt.pick_Wnormalized_cleaned_lc(kicdir,T0,W,check=True,tag="KIC"+str(kicint),savedir=args.f[0].replace(".txt",""))
        if len(lcsw[lcsw==lcsw])>0 and precw:
            lcsallw.append(lcsw)
            print("2. WNORMALIZED CLEANED LC IS APPENDED")

    np.savez(args.f[0].replace(".txt","")+"_picktrap",lcsall)
    np.savez(args.f[0].replace(".txt","")+"_picktrapW",lcsallw)
        
