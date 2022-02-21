import numpy as np
def bkgdlc(cnts,apbkg):

    nt=cnts.shape[0]
    nx=cnts.shape[1]
    ny=cnts.shape[2]
    cntsf=cnts.reshape(nt,nx*ny)
    maskbkg=apbkg.reshape(nx*ny).astype(np.bool)
    bkg=np.sum(cntsf[:,maskbkg],axis=1)

    return bkg
