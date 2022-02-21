import sys
import os
sys.path.append('D:\\SynologyDrive\\Univ\\kenkyuu\\gtrap\\gtrap')
os.environ['CPLUS_INCLUDE_PATH']=r'D:\SynologyDrive\Univ\kenkyuu\gtrap\include'
import time
import matplotlib.pyplot as plt
import math
import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda
import pycuda.compiler
from pycuda.compiler import SourceModule
import geebls
import gfilter
import getstat
import read_tesslc as tes
from astropy.io import fits
import csv

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


def bls(sector,tic,writer):
    dirlist=[filename(sector,tic)]
    lc,tu,n,ntrue,nq,inval, bjdoffset,t0arr, t, det, info=tes.load_tesslc(dirlist)
    lc=2.0-lc
    lc=lc[tu!=[inval]]
    tu=tu[tu!=[inval]]
    lc=lc[tu!=[0.]]
    tu=tu[tu!=[0.]]
    lc=lc.reshape([len(lc),1])
    tu=tu.reshape([len(tu),1])
    n=len(lc)
    ntrue=np.array([n])
    if n<=500:
        return
    offset,imgout=np.mean(lc,axis=1),lc #???
    qmi = 0.01
    qma = 0.1
    max_period=10.0
    min_period=0.2 
    fmin=(1./(float(max_period)))                                          
    df = 0.00003                                                                 
    nf =  int((1./min_period)/df)
    nb = 1024
    min_duration_hours = 0.05
    max_duration_hours = 1.0
    qmi = float(min_duration_hours)/24./max_period
    qma = float(max_duration_hours)/24./min_period                                                 
    nb = 1024
    tu=np.array(tu,order="C").astype(np.float32)
    start = time.time()
    dev_offset = cuda.mem_alloc(offset.astype(np.float32).nbytes)
    cuda.memcpy_htod(dev_offset,offset.astype(np.float32))
    
    dev_imgout=cuda.mem_alloc(imgout.astype(np.float32).nbytes)
    cuda.memcpy_htod(dev_imgout,imgout.astype(np.float32))
    
    dev_ntrue = cuda.mem_alloc(ntrue.nbytes)
    cuda.memcpy_htod(dev_ntrue,ntrue)
    
    dev_tu = cuda.mem_alloc(tu.astype(np.float32).nbytes)
    cuda.memcpy_htod(dev_tu,tu.astype(np.float32))
    
    blsp=np.zeros(nf*nq).astype(np.float32)
    dev_blsp = cuda.mem_alloc(blsp.nbytes)
    
    phase1=np.zeros(nf*nq).astype(np.uint32)
    dev_phase1 = cuda.mem_alloc(phase1.nbytes) 
    phase2=np.zeros(nf*nq).astype(np.uint32)
    dev_phase2 = cuda.mem_alloc(phase2.nbytes)  
    source_module=geebls.gbls_module()
    nthread=512
    pkernel=source_module.get_function("geebls")
    
    kma=int(qma*nb)+1
    kmi=int(qmi*nb)
    kkmi=int(n*qmi)
    sharedsize=int(4*(nb+kma)*2+4*nb)
    pkernel(dev_blsp,dev_phase1,dev_phase2,dev_imgout,dev_tu,dev_offset,dev_ntrue,\
            np.uint32(n),np.uint32(nf),np.uint32(nb),\
            np.uint32(kma),np.uint32(kmi),np.uint32(kkmi),\
            np.float32(fmin),np.float32(df),\
            block=(nthread,1,1), grid=(int(nf),int(nq)),shared=sharedsize)
    Pest,sde,phasebest1,phasebest2=getstat.get_blsstat(dev_blsp,dev_phase1,dev_phase2,nf,nq,df,fmin)
    pphase1=Pest*phasebest1/nb
    pphase2=Pest*phasebest2/nb
    duration=(pphase2-pphase1)*24
    if sde>=8 and duration>=1:
        print(str(tic)+' is a candidate of TCE. Params (Pest,sde,phasebest1,nb,pphase1,pphase2) are '+str(Pest)+','+str(sde)+','+str(phasebest1)+','+str(nb)+','+str(pphase1)+','+str(pphase2))
        writer.writerow([tic,sector,Pest[0],sde[0],phasebest1[0],nb,pphase1[0],pphase2[0]])
        fig = plt.figure(figsize=(10,10))
        for j in range(0,np.min([nq,5])):
            ax=fig.add_subplot(np.min([nq,5]),1,j+1)
            ax.plot(np.mod(tu[:,j],Pest[j]), imgout[:,j],".",markersize=3)
            ax.plot([pphase1[j]],1,"^",color="black")
            ax.plot([pphase2[j]],1,"^",color="red")
            plt.xlim(pphase1[j]-3,pphase2[j]+4)
        plt.xlabel("t")
        plt.savefig(os.path.join(os.getcwd(),"s"+str(sector).zfill(4),str(tic)+".png"))
        plt.clf()
        plt.close()

    dev_offset.free()
    dev_imgout.free()
    dev_ntrue.free()
    dev_tu.free()
    dev_blsp.free()
    dev_phase1.free()
    dev_phase2.free()


if __name__ == '__main__':
    print('start process')
    sys.path.append('D:\\SynologyDrive\\Univ\\kenkyuu\\gtrap\\gtrap')
    os.environ['CPLUS_INCLUDE_PATH']=r'D:\SynologyDrive\Univ\kenkyuu\gtrap\include'
    args=sys.argv
    sector=int(args[1])
    print('input sector is '+str(sector))
    tic_csv_name=args[2]
    print('reading ' +tic_csv_name)
    output_csv_name=args[3]
    print('output will be written to '+output_csv_name)
    output_csv=open(output_csv_name,'w')
    writer=csv.writer(output_csv)
    writer.writerow(['TIC','sector','Pest','sde','phasebest1','nb','pphase1','pphase2'])
    with open(tic_csv_name) as f:
        reader=csv.reader(f)
        l=[row for row in reader]
        numtic=len(l)-1
        print('process '+str(numtic)+' stars')
        for i in range(1,len(l)):
            print('processing '+str(i)+'/'+str(numtic))
            bls(sector,int(l[i][0]),writer)
    output_csv.close()
