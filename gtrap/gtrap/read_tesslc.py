#!/usr/bin/python
import sys
import argparse
import numpy as np
from astropy.io import fits
import glob
import re
import os



def load_tesslc(filelist,offt="t[0]",nby=1000):
    
    inval=-1.0 #invalid time
    nq=1 # # of bacth
    lc=[]
    tu=[]
    ntrue=[]
    t0arr=[]
    nfile=len(filelist)

    
    for k in range(0,nq):
        t, det, err, cno, bjdoffset, sector, info=read_tesslc(filelist)

        n=int(cno[-1]-cno[0]+1)
        while np.mod(n,nby)>0:
            n=n+1
        
        t0arr.append(t[0])
        
        lcn, tun=throw_tessintarray(n,cno,t,det,fillvalv=1.002,fillvalt=inval,offt=offt)        
        gapmask=(tun<0)
        ntrue.append(len(tun[~gapmask]))
        Hfill=np.max(lcn)
        Lfill=np.min(lcn)
        maskL=gapmask[::2]
        maskH=gapmask[1::2]   
        lcn[::2][maskL]=Lfill
        lcn[1::2][maskH]=Hfill
        tu.append(tun)
        lc.append(lcn)
    lc=np.array(lc).transpose().astype(np.float32)
    tu=np.array(tu).transpose().astype(np.float32)
    ntrue=np.array(ntrue).astype(np.uint32)

    ##original masked data
    mask=(t==t)
    t=t[mask]
    det=det[mask]
    return lc,tu,n,ntrue,nq,inval, bjdoffset,t0arr, t, det, info


def numericalSort(value):
    numbers = re.compile(r'(\d+)')
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts

def get_primaryinfo(dir="./",filename="any",lctype="llc"):
        #obj=hdulist['PRIMARY'].header["OBJECT"]        
    if filename=="any":
        fitsf=sorted(glob.glob(dir+'/*_'+lctype+'.fits'), key=numericalSort)[0]
    else:
        fitsf=filename[0]

    hdulist = fits.open(fitsf)
    head=hdulist['PRIMARY'].header

    return head

#        hdulist.info()


def read_tesslc(filelist,ticdir="", fluxtag="KSPSAP_FLUX",lctype="lc",disp=True):

    det=[]
    err=[]
    time=[]
    cno=[]
    sector=[]
    info=[]
    bjdoffset=0
    fitsf=filelist

    for efile in fitsf:
        hdulist = fits.open(os.path.join(ticdir,efile))
        bjdoffset=hdulist['LIGHTCURVE'].header["BJDREFI"]
        timetmp=hdulist['LIGHTCURVE'].data.field("TIME")
        dettmp=hdulist['LIGHTCURVE'].data.field(fluxtag)
        errtmp=hdulist['LIGHTCURVE'].data.field(fluxtag+"_ERR")
        cnotmp=hdulist['LIGHTCURVE'].data.field("CADENCENO")
        qtmp=hdulist['LIGHTCURVE'].data.field("QUALITY")
        bkgtmp=hdulist['LIGHTCURVE'].data.field("SAP_BKG")
        bkgerrtmp=hdulist['LIGHTCURVE'].data.field("SAP_BKG_ERR")

        hdulist.close()
        mask = (dettmp == dettmp)
        meanlc=np.median(dettmp[mask])
        dettmp = dettmp/meanlc
        errtmp=errtmp/meanlc
        ntmp=len(dettmp)
        
        det=np.hstack([det,dettmp])
        err=np.hstack([err,errtmp])
        time=np.hstack([time,timetmp])

        time[qtmp>0]=-1.0 #inval

        cno=np.hstack([cno,cnotmp])
        sectortmp=hdulist['PRIMARY'].header["SECTOR"]
        sector=np.hstack([sector,np.ones(ntmp,dtype=int)*sectortmp])

        #info        
        info.append(hdulist['PRIMARY'].header)
        
        
    return time, det, err, cno, bjdoffset, sector, info

def throw_tessintarray(n,cno,t,lc,fillvalv=-1.0,fillvalt=-5.0,offt="t[0]"):
    #dt=t[2]-t[1]
    #offt=t[1]
    offset=np.array(cno[0])
    jend=int(cno[-1]-offset+1)
    lcn=np.ones(n)*fillvalv
    if(jend > n):
        print("Set larger n than ",jend)
        sys.exit("Error")
#    else:
#        print("Filling ",jend,"-values in ",n," elements in lcn.")
    tun=np.ones(n)*fillvalt
    if offt=="t[0]":
        t0=t[0]
    else:
        t0=0.0

    for i in range(0,len(cno)):
        j=int(cno[i]-offset)
        if lc[i]==lc[i]:    
            lcn[j]=lc[i]
            tun[j]=t[i]-t0

    return lcn,tun



def read_vizier_fitsfile(infile):
    print ("READ VIZIER FITS",infile)
    hduread= fits.open(infile)    
    data=hduread[1].data
    ntag=hduread[1].header["TFIELDS"]
    taglist=[]

    for itag in range(0,ntag):
        taglist.append(hduread[1].header["TTYPE"+str(itag+1)].strip())

    return data, taglist

def kicdir(kicnum,ddir="/sharksuck/kic/data/"):
    #convert kic number (int) to directry name

    strk=str(kicnum)
    if kicnum>99999999:        
        rawdir=strk[0:4]+"/"+strk
    elif kicnum>9999999:        
        rawdir="0"+strk[0:3]+"/0"+strk
    else:
        rawdir="00"+strk[0:2]+"/00"+strk
    return ddir+rawdir






if __name__ == "__main__":
    print("read_keplerlc")
