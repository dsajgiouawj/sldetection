import pycuda.driver as cuda
import pycuda.autoinit
import pycuda.compiler
from pycuda.compiler import SourceModule
import numpy as np
import time

def gmf_module():
    #median filter
    source_module = SourceModule("""
    #include <stdio.h>
    #define FILVAL 0.0
    __global__ void gfilter(float *input, float *output,int i_dim, int j_dim, int r){
    
    int idc, inf, equal, sup;
    float val, min, max, mxinf, minsup, estim;
    
    int ib = threadIdx.y;
    int jb = threadIdx.x;
    int j = blockIdx.x*blockDim.x + jb;
    int i = blockIdx.y*blockDim.y + ib;    
    extern __shared__ float buff[];
 
    /* fortran-wise */
    int offset = blockDim.x*r;
    int idx_h = (ib+r)*blockDim.x + jb;
    buff[idx_h] = input[i*j_dim + j];
    if (ib < r && i-r >= 0){
    buff[idx_h - offset] = input[(i-r)*j_dim + j];
    }else if(ib < r){
    buff[idx_h - offset] = FILVAL;
    }else if (ib >= (blockDim.y-r) && i+r < i_dim){
    buff[idx_h + offset] = input[(i+r)*j_dim + j];
    }else if (ib >= (blockDim.y-r) && i+r >= i_dim){
    buff[idx_h + offset] = FILVAL;
    }

    __syncthreads(); 
    
    min = max = buff[ib*blockDim.x+jb];
    for ( idc=0; idc<2*r+1; idc++){
    val = buff[(ib+idc)*blockDim.x+jb];
    if ( val < min ) min = val;
    if ( val > max ) max = val;
    }
    
    int it = 0;
    while (1){
        estim = (min+max)/2;
        inf = sup = equal = 0;
        mxinf = min;
        minsup = max;
        for (idc =0; idc < 2*r+1; idc++){
            val = buff[(ib+idc)*blockDim.x+jb];
            if (val < estim ){
                inf++;
                if( val > mxinf ) mxinf = val;
            } else if (val > estim) {
                sup++;
                if( val < minsup) minsup = val;
            } else equal++; 
        }
        if ( (inf <= (r+1))&&(sup <= (r+1))) break;
        else if (inf>sup) max = mxinf;
        else min = minsup;
        it++;
        if(it > 2*r){
        val = 0;
        break;
    }
    }
    if (inf >= r+1 ) val = mxinf;
    else if (inf+equal >= r+1) val = estim;
    else val = minsup;

    val = buff[idx_h] - val;

/* Fortran-wise
/*
    output[j*i_dim + i] = val; 
*/
/* C-wise */

    output[i*j_dim + j] = val; 

    }
    
""",options=['-use_fast_math'])
    return source_module

def maxpow(num):
    return 2**int(np.ceil(np.log2(num)))

def get_offset(n,nq,dev_imgout,tu):
    imgout= np.zeros(nq*n).astype(np.float32)
    cuda.memcpy_dtoh(imgout, dev_imgout)
    imgout[tu.flatten()<0.0]=0.0
    imgout=imgout.reshape((n,nq))
    offset=np.mean(imgout,axis=0)

    return offset,imgout

def get_detrend(img,nby,nbx,r,isw=0,osw=0):
    start = time.time()

    if (not np.isfortran(img)):
        print("Warning: gfilter assumes a Fortran-wise batch array. Probably, wrong results?")

    xdim=np.shape(img)[1]
    ydim=np.shape(img)[0]
    print("ydim=",ydim,"xdim=",xdim)

    if int(xdim/nbx)!=xdim/nbx:
        print("Warning!: xdim should be divisible by nbx.")                
    if int(ydim/nby)!=ydim/nby:
        print("Warning!: ydim should be divisible by nby.")

    sharedsize=int(4*(2*r+nby)*nbx)
    #read kernel
    source_module=gmf_module()
#    gkernel = source_module.get_function("gmedfilter")
    gkernel = source_module.get_function("gfilter")
    #load
    if isw==0:
        dev_in=cuda.mem_alloc(img.nbytes)
    cuda.memcpy_htod(dev_in, img.flatten().astype(np.float32)) 
    
    #global memory
    if isw==0:
        dev_out=cuda.mem_alloc(img.nbytes)
    gkernel(dev_in,dev_out,np.uint32(ydim),np.uint32(xdim),np.uint32(r),block=(nbx,nby,1),grid=(int(xdim/nbx),int(ydim/nby)),shared=sharedsize)

    if osw==0:
        out1= np.zeros(xdim*ydim).astype(np.float32)
        cuda.memcpy_dtoh(out1, dev_out)         

    elapsed_time = time.time() - start
    print(("elapsed_time:{0}".format(elapsed_time)) + "[sec]")

    if osw==0:
        return out1.reshape((ydim,xdim))
    else:
        return dev_out
        #    return img - out1.reshape((xdim,ydim)).transpose()

if __name__ == "__main__":
    print("gfilter")
