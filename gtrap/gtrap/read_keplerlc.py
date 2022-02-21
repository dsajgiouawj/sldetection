#!/usr/bin/python
import sys
import argparse
import numpy as np
from astropy.io import fits
import glob
import re

def kicdir(kicnum,ddir="/sharksuck/kic/data/"):
    #convert kic number (int) to directry name
    strk=str(kicnum)
    if kicnum>99999999:        
        rawdir=strk[0:4]+"/"+strk
    elif kicnum>9999999:        
        rawdir="0"+strk[0:3]+"/0"+strk
    elif kicnum>999999:
        rawdir="00"+strk[0:2]+"/00"+strk        
    else:
        rawdir="000"+strk[0:1]+"/000"+strk
    return ddir+rawdir


def load_keplc(dirlist,offt="t[0]"):
    if len(dirlist) == 0:
        sys.exit("No Directory to read in load_keplc. Exit.")
    n=72000
    inval=-1000.0 #invalid time
    nq=len(dirlist) # # of bacth
    lc=[]
    tu=[]
    ntrue=[]
    t0arr=[]
    for k in range(0,nq):
        print(dirlist)
        t, det, err, cno, bjdoffset, quarter, season=read_lc(dirlist[k])
        t0arr.append(t[0])
        
        lcn, tun=throw_kepintarray(n,cno,t,det,fillvalv=1.002,fillvalt=inval,offt=offt)        
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
    return lc,tu,n,ntrue,nq,inval, bjdoffset,t0arr, t, det


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


def read_lc(dir="./",filename="all", fluxtag="PDCSAP_FLUX",lctype="llc",disp=True):
    det=[]
    err=[]
    time=[]
    cno=[]
    quarter=[]
    season=[]
    bjdoffset=0
    if filename=="all":
        fitsf=sorted(glob.glob(dir+'/*_'+lctype+'.fits'), key=numericalSort)
    else:
        fitsf=filename

    for file in fitsf:
        if filename=="all":
            hdulist = fits.open(file)
        else:
            hdulist = fits.open(dir+file)
#        hdulist.info()
        bjdoffset=hdulist['LIGHTCURVE'].header["BJDREFI"]
        timetmp=hdulist['LIGHTCURVE'].data.field("TIME")
        dettmp=hdulist['LIGHTCURVE'].data.field(fluxtag)
        errtmp=hdulist['LIGHTCURVE'].data.field(fluxtag+"_ERR")
        cnotmp=hdulist['LIGHTCURVE'].data.field("CADENCENO")
        hdulist.close()
        mask = (dettmp == dettmp)
        meanlc=np.median(dettmp[mask])
        dettmp = dettmp/meanlc
        errtmp=errtmp/meanlc
        ntmp=len(dettmp)
        
        det=np.hstack([det,dettmp])
        err=np.hstack([err,errtmp])
        time=np.hstack([time,timetmp])
        cno=np.hstack([cno,cnotmp])
        quartertmp=hdulist['PRIMARY'].header["QUARTER"]
        seasontmp=hdulist['PRIMARY'].header["SEASON"]
        quarter=np.hstack([quarter,np.ones(ntmp,dtype=int)*quartertmp])
        season=np.hstack([season,np.ones(ntmp,dtype=int)*seasontmp])

    return time, det, err, cno, bjdoffset, quarter, season

def throw_kepintarray(n,cno,t,lc,fillvalv=-1.0,fillvalt=-5.0,offt="t[0]"):
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
