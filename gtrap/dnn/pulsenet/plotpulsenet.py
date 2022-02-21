import matplotlib.pyplot as plt
import numpy as np

#        ndmax = ai.compute_ndmax(cnts)
#        icb1,icb2  = icb.compute_icb(cnts,apbkg)
# T0j=T0[j]

def plotBC(time,flux,q,icb1,icb2, ndmax,T0j, title,out):
    fig = plt.figure()
    ax = fig.add_subplot(311)
    mask=(q==0)
    plt.axvline(T0j,color="green",ls="dashed",alpha=0.5)
    ax.plot(time[mask],flux[mask],".")
    plt.title(title)
    ax = fig.add_subplot(312)
    ax.plot(time[mask],ndmax[mask])
    plt.axvline(T0j,color="green",ls="dashed",alpha=0.5)
    ax = fig.add_subplot(313)
    ax.plot(time[mask],icb1[mask])
    ax.plot(time[mask],icb2[mask])
    plt.axvline(T0j,color="green",ls="dashed",alpha=0.5)
    plt.savefig(out)
    plt.close()
    
