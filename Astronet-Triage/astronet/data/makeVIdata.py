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


def convert(tic_id,score,Epoc,Period,Duration,Sectors,output_dir):
    dirlist=[filename(Sectors,tic_id)]
    lc,tu,n,ntrue,nq,inval, bjdoffset,t0arr, t, det, info=tes.load_tesslc(dirlist)
    lc=lc[tu!=[inval]]
    tu=tu[tu!=[inval]]
    lc=lc[tu!=[0.]]
    tu=tu[tu!=[0.]]
    lc=lc.reshape([len(lc),1])
    tu=tu.reshape([len(tu),1])
    if 1.3<np.max(lc):
        return
    Epoc=Epoc-t0arr[0]
    pphase12=2*Epoc #pphase1+pphase2
    pphase2_1=Duration/24#pphase2-pphase1
    pphase2=(pphase12+pphase2_1)/2
    pphase1=pphase12-pphase2
    # Epoc->Period/2
    tu=tu-(Epoc-Period/2)
    fig=plt.figure(figsize=(35,10))
    ax0=fig.add_subplot(1,3,1)
    ax0.plot(tu[:,0],lc[:,0],".",markersize=3)
    ax=fig.add_subplot(1,3,2)
    ax.plot(np.mod(tu[:,0],Period),lc[:,0],".",markersize=3)
    ax2=fig.add_subplot(1,3,3)
    ax2.plot(np.mod(tu[:,0],Period),lc[:,0],".",markersize=3)
    ax2.set_xlim(Period/2-Duration/24*2,Period/2+Duration/24*2)
    plt.xlabel(score)
    plt.savefig(os.path.join(output_dir,"s"+str(Sectors).zfill(3)+"_"+str(tic_id)+".png"))
    plt.clf()
    plt.close()


if __name__ == '__main__':
    print('start process')
    sys.path.append('D:\\SynologyDrive\\Univ\\kenkyuu\\gtrap\\gtrap')
    args=sys.argv
    tic_csv_name=args[1]
    print('reading ' +tic_csv_name)
    output_dir=args[2]
    print('output_dir is '+output_dir)

    with open(tic_csv_name) as f:
        reader=csv.reader(f)
        l=[row for row in reader]
        numtic=len(l)-1
        print('process '+str(numtic)+' stars')
        for i in range(1,len(l)):
            print('processing '+str(i)+'/'+str(numtic))
            convert(tic_id=int(l[i][0]),score=float(l[i][1]),Epoc=float(l[i][2]),Period=float(l[i][3]),Duration=float(l[i][4]),Sectors=int(l[i][5]),output_dir=output_dir)
