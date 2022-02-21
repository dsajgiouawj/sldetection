def pick_cleaned_lc(kicdir,T0,wid=128,contcrit=84,check=False,lcout=False,tag="",savedir="./"):
    lc,tu,n,ntrue,nq,inval,bjdoffset,t0, t,det=kep.load_keplc([kicdir],offt="0")
    mask=(tu>0.0)
    tuu=tu[mask]
    i=np.searchsorted(tuu,T0)
    numar=np.array(range(0,len(tu)))
    ii=numar[tuu[i]==tu[:,0]][0]

    istart=ii-wid
    iend=ii+wid
    tus=np.copy(tu[istart:iend,0])
    lcs=np.copy(lc[istart:iend,0])

    #pre classifier (check continuous null region)
    prec = True 
    if len(tus) == 0:
        prec=False
    elif tus[0] < 0.0 or tus[-1]<-1:
        contsw=0
        for j, teach in enumerate(tus):
            if teach < 0.0:
                contsw=contsw+1
                if contsw > contcrit:
                    prec=False
            else:
                contsw=0
        
    
    #### CLEAN LCS UP ####
    sw=0
    
    while sw==0:
        tusplus=np.concatenate([[-2000],tus[:-1]])
        tusminus=np.concatenate([tus[1:],[-2000]])
        lcsplus=np.concatenate([[0],lcs[:-1]])
        lcsminus=np.concatenate([lcs[1:],[0]])
        medv=np.median(lcs)

        for j, teach in enumerate(tus):
            if teach < 0.0:
                if tusminus[j] > 0.0 and tusplus[j] > 0.0:
#                    lcs[j] = (lcsminus[j])
                    lcs[j] = (lcsminus[j] + lcsplus[j])/2.0
                    tus[j]=1000.0

                elif tusminus[j] > 0.0:
                    lcs[j] = medv                    
#                    lcs[j] = lcsminus[j]
                    tus[j]=1000.0
                elif tusplus[j] > 0.0:
                    lcs[j] = medv                    
#                    lcs[j] = lcsplus[j]
                    tus[j]=1000.0

        if len(tus[tus<0.0])==0:
            sw = 1
    ### median filter
    lcs=signal.medfilt(lcs,kernel_size=3)

    ### NORMALIZE
    lcs=(lcs-np.mean(lcs))/np.std(lcs)
    
    if check:
        fig=plt.figure()
        ax=fig.add_subplot(211)
        istartx=np.max([0,istart-wid])
        iendx=np.min([len(lc),iend+wid])
        lctmp=lc[istartx:iendx]
        tutmp=tu[istartx:iendx]
        mask=(tutmp>0.0)
        ax.plot(tutmp[mask],lctmp[mask],".",color="gray",label="before clean")

        plt.legend()
        ax=fig.add_subplot(212)
        ax.plot(lcs,".",color="orange",label="cleaned vector")
        plt.legend()
        plt.savefig(os.path.join(savedir,tag+"vector.png"))
        #        plt.show()
        
    if lcout:
        return lcs, tus, lc[istart:iend,0], tu[istart:iend,0], lc[:,0], tu[:,0], prec
    else:
        return lcs, tus, prec


def pick_Wnormalized_cleaned_lc(kicdir,T0,W,alpha=2,nx=128,daytopix=48,contcrit=84,check=False,lcout=False,tag="",savedir="./"):
    #nx=64 length of WNC vector
    lc,tu,n,ntrue,nq,inval,bjdoffset,t0, t,det=kep.load_keplc([kicdir],offt="0")
    mask=(tu>0.0)
    tuu=tu[mask]
    i=np.searchsorted(tuu,T0)
    numar=np.array(range(0,len(tu)))
    ii=numar[tuu[i]==tu[:,0]][0]


    wid=int(alpha*W*daytopix)
    print("The range is between -",alpha," W to +",alpha," W." )
    istart=max(0,ii-wid)
    iend=min(ii+wid,len(tu))
    tus=np.copy(tu[istart:iend,0])
    lcs=np.copy(lc[istart:iend,0])
    
    #pre classifier (check continuous null region)
    prec = True 
    if len(tus) == 0:
        prec=False
    elif tus[0] < 0.0 or tus[-1]<-1:
        contsw=0
        for j, teach in enumerate(tus):
            if teach < 0.0:
                contsw=contsw+1
                if contsw > contcrit:
                    prec=False
            else:
                contsw=0
        
    
    #### CLEAN LCS UP ####
    sw=0
    
    while sw==0:
        tusplus=np.concatenate([[-2000],tus[:-1]])
        tusminus=np.concatenate([tus[1:],[-2000]])
        lcsplus=np.concatenate([[0],lcs[:-1]])
        lcsminus=np.concatenate([lcs[1:],[0]])
        medv=np.median(lcs)

        for j, teach in enumerate(tus):
            if teach < 0.0:
                if tusminus[j] > 0.0 and tusplus[j] > 0.0:
#                    lcs[j] = (lcsminus[j])
                    lcs[j] = (lcsminus[j] + lcsplus[j])/2.0
                    tus[j]=1000.0

                elif tusminus[j] > 0.0:
                    lcs[j] = medv                    
#                    lcs[j] = lcsminus[j]
                    tus[j]=1000.0
                elif tusplus[j] > 0.0:
                    lcs[j] = medv                    
#                    lcs[j] = lcsplus[j]
                    tus[j]=1000.0

        if len(tus[tus<0.0])==0:
            sw = 1
    ### median filter
    lcs=signal.medfilt(lcs,kernel_size=3)

    ### NORMALIZE
    lcs=(lcs-np.mean(lcs))/np.std(lcs)
    nlcs=len(lcs)
    tt=np.array(range(0,nlcs))*2.0*alpha/nlcs - alpha
    fx = interpolate.interp1d(tt, lcs)
    tx=np.array(range(0,nx))*2.0*alpha/nx - alpha
    lcsx=fx(tx)
    
    if check:
        fig=plt.figure()
        ax=fig.add_subplot(211)
        istartx=np.max([0,istart-wid])
        iendx=np.min([len(lc),iend+wid])
        lctmp=lc[istartx:iendx]
        tutmp=tu[istartx:iendx]
        mask=(tutmp>0.0)
        ax.plot(tutmp[mask],lctmp[mask],".",color="gray",label="before clean")

        plt.legend()
        ax=fig.add_subplot(212)
        ax.plot(tt,lcs,".",color="orange")
        ax.plot(tx,lcsx,".",color="red")

        plt.xlabel("time (W)")
        plt.ylabel("WNC vector")
        plt.savefig(os.path.join(savedir,tag+"vector_w.png"))
        #        plt.show()
        
    if lcout:
        return lcs, tus, lc[istart:iend,0], tu[istart:iend,0], lc[:,0], tu[:,0], prec
    else:
        return lcsx, tx, prec


def pick_cleaned_lc_direct(lc,tu,T0,wid=128,contcrit=84,check=False,lcout=False,tag="",savedir="./"):
#    lc,tu,n,ntrue,nq,inval,bjdoffset,t0, t,det=kep.load_keplc([kicdir],offt="0")
    mask=(tu>0.0)
    tuu=tu[mask]
    i=np.searchsorted(tuu,T0)
    numar=np.array(range(0,len(tu)))
    ii=numar[tuu[i]==tu[:,0]][0]

    istart=ii-wid
    iend=ii+wid+1
    tus=np.copy(tu[istart:iend,0])
    lcs=np.copy(lc[istart:iend,0])

    #pre classifier (check continuous null region)
    prec = True 
    if len(tus) == 0:
        prec=False
    elif tus[0] < 0.0 or tus[-1]<-1:
        contsw=0
        for j, teach in enumerate(tus):
            if teach < 0.0:
                contsw=contsw+1
                if contsw > contcrit:
                    prec=False
            else:
                contsw=0
        
    
    #### CLEAN LCS UP ####
    sw=0
    
    while sw==0:
        tusplus=np.concatenate([[-2000],tus[:-1]])
        tusminus=np.concatenate([tus[1:],[-2000]])
        lcsplus=np.concatenate([[0],lcs[:-1]])
        lcsminus=np.concatenate([lcs[1:],[0]])
        medv=np.median(lcs)

        for j, teach in enumerate(tus):
            if teach < 0.0:
                if tusminus[j] > 0.0 and tusplus[j] > 0.0:
#                    lcs[j] = (lcsminus[j])
                    lcs[j] = (lcsminus[j] + lcsplus[j])/2.0
                    tus[j]=1000.0

                elif tusminus[j] > 0.0:
                    lcs[j] = medv                    
#                    lcs[j] = lcsminus[j]
                    tus[j]=1000.0
                elif tusplus[j] > 0.0:
                    lcs[j] = medv                    
#                    lcs[j] = lcsplus[j]
                    tus[j]=1000.0

        if len(tus[tus<0.0])==0:
            sw = 1
    ### median filter
    lcs=signal.medfilt(lcs,kernel_size=3)

    ### NORMALIZE
    lcs=(lcs-np.mean(lcs))/np.std(lcs)
    
    if check:
        fig=plt.figure()
        ax=fig.add_subplot(211)
        istartx=np.max([0,istart-wid])
        iendx=np.min([len(lc),iend+wid])
        lctmp=lc[istartx:iendx]
        tutmp=tu[istartx:iendx]
        mask=(tutmp>0.0)
        ax.plot(tutmp[mask],lctmp[mask],".",color="gray",label="before clean")

        plt.legend()
        ax=fig.add_subplot(212)
        ax.plot(lcs,".",color="orange",label="cleaned vector")
        plt.legend()
        plt.savefig(os.path.join(savedir,tag+"vector.png"))
        #        plt.show()
        
    if lcout:
        return lcs, tus, lc[istart:iend,0], tu[istart:iend,0], lc[:,0], tu[:,0], prec
    else:
        return lcs, tus, prec
