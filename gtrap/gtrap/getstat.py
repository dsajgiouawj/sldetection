import time
import matplotlib.pyplot as plt
import math
import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda
import pycuda.compiler
from pycuda.compiler import SourceModule

def gstat_module():
    source_module = SourceModule("""

    #include <stdio.h>
    __global__ void getmax(float *blsp, unsigned int *phase1, unsigned int *phase2, float *maxval, unsigned int *maxidx, float *avesum, float *varsum, unsigned int *maxphase1, unsigned int *maxphase2){
    unsigned int nthread = blockDim.x;
    unsigned int ithread = threadIdx.x;
    extern __shared__ float cache[]; 

    /*
    patch number = blockIdx.x
    batch number = blockIdx.y
    nf = nthread * gridDim.x
    */

    /* thread cooperating load */    
    unsigned int i = ithread+blockIdx.x*nthread+nthread*gridDim.x*blockIdx.y;
    float val = blsp[i];
    cache[ithread]=val;
    cache[ithread+nthread]=val;
    cache[ithread+2*nthread]=val*val;
    __syncthreads();

    unsigned int k = nthread/2;

    while (k !=0){                                                                       
    if (ithread < k){
    cache[ithread] = max(cache[ithread],cache[ithread+k]);
    cache[ithread+nthread] = cache[ithread+nthread]+cache[ithread+nthread+k];
    cache[ithread+2*nthread] = cache[ithread+2*nthread]+cache[ithread+2*nthread+k];
    }
    __syncthreads();
    k /= 2;                                                                         
    }
    __syncthreads();                                                                     

    if (val==cache[0]){
    maxval[blockIdx.x+gridDim.x*blockIdx.y] = cache[0]; 
    maxidx[blockIdx.x+gridDim.x*blockIdx.y] = i; 
    avesum[blockIdx.x+gridDim.x*blockIdx.y] = cache[nthread]; 
    varsum[blockIdx.x+gridDim.x*blockIdx.y] = cache[2*nthread]; 
    maxphase1[blockIdx.x+gridDim.x*blockIdx.y] = phase1[i];
    maxphase2[blockIdx.x+gridDim.x*blockIdx.y] = phase2[i];

    }


    }


    """,options=['-use_fast_math'])

    return source_module

def get_blsstat(dev_blsp,dev_phase1,dev_phase2,nf,nq,df,fmin,nthread=1024):
    maxval,maxidx,avesum,varsum,maxphase1,maxphase2,npatch=patch_stat(dev_blsp,dev_phase1,dev_phase2,nf,nq,nthread)
    Pest, sdearr, phasebest1, phasebest2=combine_stat(maxval,maxidx,avesum,varsum,maxphase1,maxphase2,npatch,nq,nf,df,fmin)
    return Pest, sdearr, phasebest1,phasebest2
    
def patch_stat(dev_blsp,dev_phase1,dev_phase2,nf,nq,nthread=1024):
    source_module=gstat_module()
    kernel=source_module.get_function("getmax")
    sharedsize=int(4*nthread*3)
    npatch=int(nf/nthread)
    maxval= np.zeros(npatch*nq).astype(np.float32)
    maxidx= np.zeros(npatch*nq).astype(np.uint32)
    avesum= np.zeros(npatch*nq).astype(np.float32)
    varsum= np.zeros(npatch*nq).astype(np.float32)
    maxphase1= np.zeros(npatch*nq).astype(np.uint32)
    maxphase2= np.zeros(npatch*nq).astype(np.uint32)

    dev_maxval=cuda.mem_alloc(maxval.nbytes)
    dev_maxidx=cuda.mem_alloc(maxidx.nbytes)
    dev_avesum=cuda.mem_alloc(avesum.nbytes)
    dev_varsum=cuda.mem_alloc(varsum.nbytes)
    dev_maxphase1=cuda.mem_alloc(maxphase1.nbytes)
    dev_maxphase2=cuda.mem_alloc(maxphase2.nbytes)

    kernel(dev_blsp,dev_phase1,dev_phase2,dev_maxval,dev_maxidx,\
           dev_avesum,dev_varsum,dev_maxphase1, dev_maxphase2,\
           block=(nthread,1,1), grid=(int(npatch),int(nq)),shared=sharedsize)
    cuda.memcpy_dtoh(maxval, dev_maxval)
    cuda.memcpy_dtoh(maxidx, dev_maxidx)
    cuda.memcpy_dtoh(avesum, dev_avesum)
    cuda.memcpy_dtoh(varsum, dev_varsum)
    cuda.memcpy_dtoh(maxphase1, dev_maxphase1)
    cuda.memcpy_dtoh(maxphase2, dev_maxphase2)

    return maxval,maxidx,avesum,varsum,maxphase1,maxphase2,npatch

def combine_stat(maxval,maxidx,avesum,varsum,maxphase1,maxphase2,npatch,nq,nf,df,fmin):
#    stdarr=[]
#    maxvarr=[]
    sdearr=[]
    Pest=[]
    phasebest1=[]
    phasebest2=[]

    for i in range(0,nq):
        idx_patch=np.argmax(maxval[0+i*npatch:npatch+i*npatch])
        idx=maxidx[idx_patch+i*npatch]
        Pest.append(1/((idx - nf*i)*df+fmin))
        maxv=np.max(maxval[0+i*npatch:npatch+i*npatch])
#        maxvarr.append(maxv)
        ave=np.sum(avesum[0+i*npatch:npatch+i*npatch])/nf
        var=np.sum(varsum[0+i*npatch:npatch+i*npatch])/nf
        std=np.sqrt(var-ave*ave)
#        stdarr.append(std)
        sde=(maxv - ave)/std
        sdearr.append(sde)
        phasebest1.append(maxphase1[idx_patch+i*npatch])
        phasebest2.append(maxphase2[idx_patch+i*npatch])

    return np.array(Pest), np.array(sdearr), np.array(phasebest1), np.array(phasebest2)



if __name__ == "__main__":

    print ("gpu eebls")
