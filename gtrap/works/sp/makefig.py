import matplotlib.pyplot as plt
import numpy as np
import gtrap.tls as tls

def trapfig(tlst,tlssn,tlsw,tu,tu0,llc,peak,idetect,ttc,ttcn,llcn,offsetlc,ffac,H,W,L,T0,sw,tic):
    fig=plt.figure(figsize=(10,7.5))
    ax=fig.add_subplot(3,1,1)
    plt.tick_params(labelsize=18)
    
#    plt.plot(tlst0[iq::nq]+tu0[iq],tlssn[iq::nq],color="gray")
#    plt.plot(tlst0[iq::nq][peak]+tu0[iq],tlssn[iq::nq][peak],"v",color="red",alpha=0.3)
    plt.plot(tlst+tu0,tlssn,color="gray")
    plt.plot(tlst[peak]+tu0,tlssn[peak],"v",color="red",alpha=0.3)   
#    plt.axvline(tlst,color="red",alpha=0.3)
#    plt.axhline(median,color="green",alpha=0.3)
    plt.ylabel("TLS series",fontsize=18)            
    #            plt.title("KIC"+str(kic)+" SN="+str(round(maxsn,1))+" far="+str(round(far,1))+" dp="+str(round(diff,1)))
    plt.title("TIC"+str(tic)+" (transit injected)",fontsize=18)
#    plt.axvline(t0+tu0,color="orange",ls="dotted")
    
    ax=fig.add_subplot(3,1,2)
    plt.tick_params(labelsize=18)
    
#    plt.xlim(tlst0[iq::nq][idetect-tlsw[iq::nq][idetect]*ffac+tu0[iq],tlst0[iq::nq][idetect]+tlsw[iq::nq][idetect]*ffac+tu0[iq])
    plt.xlim(tlst[idetect]-tlsw[idetect]*ffac+tu0,tlst[idetect]+tlsw[idetect]*ffac+tu0)
#    ax.plot(tu[im:ix,iq]+tu0[iq],llc,".",color="gray")
    ax.plot(tu+tu0,llc,".",color="gray")

    ##==================##
    
    mask=(ttc>0.0)
    dip=np.abs((np.max(llc[mask])-np.min(llc[mask]))/3.0)
    maskn=(ttcn>0.0)
    
    plt.ylim(np.min(llcn[maskn])-dip,np.max(llcn[maskn])+dip)
    plt.ylabel("PDCSAP",fontsize=18)
#    plt.axvline(t0,color="red")
    
    ax=fig.add_subplot(3,1,3)

    #                try:
    if True:
        print(idetect,"<==")
        pre=tls.gen_trapzoid(ttc[mask]+tu0,H,W,L,T0)
        if sw == 0:    
            pre = 1.0 - pre
        elif sw == 1:    
            pre = 1.0 + pre
            ax.plot(ttc[mask]+tu0,pre-(1.0-offsetlc),color="green")
            ax.plot(ttc[mask]+tu0,pre-(1.0-offsetlc),".",color="green",alpha=0.3)
            
            #        plt.xlim(tlst0[iq::nq][idetect]-tlsw[iq::nq][idetect]*ffac+tu0[iq],tlst0[iq::nq][idetect]+tlsw[iq::nq][idetect]*ffac+tu0[iq])
        plt.xlim(tlst[idetect]-tlsw[idetect]*ffac+tu0,tlst[idetect]+tlsw[idetect]*ffac+tu0)
        plt.ylim(np.min(llcn[maskn])-dip,np.max(llcn[maskn])+dip)

#                    plt.ylim(np.min(llcn[maskn])-dip,np.max(llcn[maskn])+dip)
        plt.ylabel("best matched",fontsize=18)
        if tu0==0.0:
            plt.xlabel("Day",fontsize=18)
        else:
            plt.xlabel("BKJD",fontsize=18)
            #                except:
            #                    print("IGNORE")
                       
#        plt.axvline(t0,color="red")
        plt.savefig("TIC"+str(tic)+".mock.png")

