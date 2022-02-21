import numpy as np
def compute_icb(cnts,apbkg):
    nt=cnts.shape[0]
    nx=cnts.shape[1]
    ny=cnts.shape[2]
    cntsf=cnts.reshape(nt,nx*ny)    
    maskbkgin=apbkg.reshape(nx*ny).astype(np.bool)

    crossbkgmask=np.array([[True,True,True,True,True,True,True,True,True,True,True,True,True],[True,True,True,True,True,True,True,True,True,True,True,True,True],[True,True,True,True,True,True,True,True,True,True,True,True,True],[True,True,True,True,True,True,True,True,True,True,True,True,True],[True,True,True,True,True,True,True,True,True,True,True,True,True],[True,True,True,True,True,True,True,True,True,True,True,True,True],[True,True,True,True,True,True,False,False,False,False,False,False,False],[False,False,False,False,False,False,False,False,False,False,False,False,False],[False,False,False,False,False,False,False,False,False,False,False,False,False],[False,False,False,False,False,False,False,False,False,False,False,False,False],[False,False,False,False,False,False,False,False,False,False,False,False,False],[False,False,False,False,False,False,False,False,False,False,False,False,False],[False,False,False,False,False,False,False,False,False,False,False,False,False]])
    
    for sw in [True, False]:
        masklc=np.zeros(nx*ny,dtype=bool)
        maskbkg=np.zeros(nx*ny,dtype=bool)
        if sw:
            crossbkgmask=(crossbkgmask).T
        cbm=crossbkgmask.reshape(nx*ny).astype(np.bool)
        icbm=np.invert(cbm)

        if len(maskbkgin[cbm])>0:
        
            for i in range(len(masklc)):
                masklc[i]=cbm[i] and maskbkgin[i]
            for i in range(len(maskbkg)):
                maskbkg[i]=icbm[i] and maskbkgin[i]

            nlc=np.sum(masklc)
            nbkg=np.sum(maskbkg)
            icb=np.sum(cntsf[:,masklc],axis=1)
            bkg=np.sum(cntsf[:,maskbkg],axis=1)

            if sw:
                icb1 = icb - bkg/nbkg*nlc
            else:
                icb2 = icb - bkg/nbkg*nlc
        else:
            if sw:
                icb1 = np.zeros(nt)
            else:
                icb2 = np.zeros(nt)
        
    return icb1,icb2
