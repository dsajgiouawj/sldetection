import numpy as np
from scipy.signal import medfilt 

def compute_ndmax(cnts):
    nt=cnts.shape[0]
    nx=cnts.shape[1]
    ny=cnts.shape[2]

    ndiff = cnts[1:,:,:]-cnts[:-1,:,:]
    ndiff = np.concatenate([np.zeros((1,nx,ny)),ndiff])
    ndiff = ndiff.reshape(nt,nx*ny)
    ndmax = np.max(np.abs(ndiff[:,:]),axis=1)
    ndmax = ndmax/medfilt(ndmax,kernel_size=51)
    return ndmax
