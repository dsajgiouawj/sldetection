import sys
import os
sys.path.append('D:\\SynologyDrive\\Univ\\kenkyuu\\gtrap\\gtrap')
import time
import matplotlib.pyplot as plt
import math
import numpy as np
import read_tesslc as tes
from astropy.io import fits
import csv
import matplotlib
import matplotlib.cm as cm

matplotlib.use("Agg")

def filename(sector,tic):
    if sector<=13:
        filename='F'
    else:
        filename='G'
    filename=filename+':\\/QLP/YEAR'
    if sector<=13:
        filename=filename+'1'
    else:
        filename=filename+'2'
    filename=filename+'/s'+str(sector).zfill(4)+'/'
    filename=filename+str(tic//1000000000000).zfill(4)+'/'+str(tic//100000000%10000).zfill(4)+'/'+str(tic//10000%10000).zfill(4)+'/'+str(tic%10000).zfill(4)+'/'
    filename=filename+'hlsp_qlp_tess_ffi_s'+str(sector).zfill(4)+'-'+str(tic).zfill(16)+'_tess_v01_llc.fits'
    return filename


def convert(tic_id,Epoc,Period,Duration,Sectors,output_dir):
    dirlist=[filename(Sectors,tic_id)]
    lc,tu,n,ntrue,nq,inval, bjdoffset,t0arr, t, det, info=tes.load_tesslc(dirlist,offt="0")
    lc=lc[tu!=[inval]]
    tu=tu[tu!=[inval]]
    lc=lc[tu!=[0.]]
    tu=tu[tu!=[0.]]
    lc=lc.reshape([len(lc),1])
    tu=tu.reshape([len(tu),1])
    #Epoc=Epoc-t0arr[0]
    pphase12=2*Epoc #pphase1+pphase2
    pphase2_1=Duration/24#pphase2-pphase1
    pphase2=(pphase12+pphase2_1)/2
    pphase1=pphase12-pphase2
    # Epoc->Period/2
    #tu=tu-(Epoc-Period/2)
    tu2=tu-(Epoc-Period/2)
    tu2=np.mod(tu2[:,0],Period)
    inpulse0l=Period/2-Duration/2/24<tu2
    inpulse0r=tu2<Period/2+Duration/2/24
    inpulse1l=Period/2-Duration/24<tu2
    inpulse1r=tu2<Period/2+Duration/24
    inpulse2=[0 for x in range(len(inpulse0l))]
    for i in range(len(inpulse0l)):
        if inpulse0l[i] and inpulse0r[i]:
            inpulse2[i]=1
        elif inpulse1l[i] and inpulse1r[i]:
            inpulse2[i]=0.5
    fig=plt.figure(figsize=(35,10))
    ax0=fig.add_subplot(1,3,1)
    ax0.scatter(tu[:,0],lc[:,0],vmin=0,vmax=1,c=inpulse2,cmap=cm.Paired)
    ax=fig.add_subplot(1,3,2)
    ax.plot(tu[:,0],lc[:,0],".",markersize=3)
    ax2=fig.add_subplot(1,3,3)
    ax2.plot(np.mod(tu[:,0],Period),lc[:,0],".",markersize=3)
    ax2.set_xlim(Period/2-Duration/24*2,Period/2+Duration/24*2)
    plt.savefig(os.path.join(output_dir,"s"+str(Sectors).zfill(3)+"_"+str(tic_id)+".png"))
    plt.clf()
    plt.close()
    print(tu[:,0])


if __name__ == '__main__':
    print('start process')
    sys.path.append('D:\\SynologyDrive\\Univ\\kenkyuu\\gtrap\\gtrap')
    args=sys.argv
    tic_id=int(args[1])
    sector=int(args[2])
    Epoc=float(args[3])
    Period=float(args[4])
    Duration=float(args[5])
    output_dir=args[6]
    print('output_dir is '+output_dir)
    convert(tic_id,Epoc,Period,Duration,sector,output_dir)
