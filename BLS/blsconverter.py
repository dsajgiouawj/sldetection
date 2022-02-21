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


def average_in_transit(time,flux,duration,period,t0):
  n=0
  sum=0
  for i in range(0,len(time)):
    t=(time[i]-t0)%period
    assert 0<=t and t<=period
    if (duration/2<=t and t<=period-duration/2) or time[i]<=0.1:
      pass
    else:
      n+=1
      sum+=flux[i]
  if n==0:
    return 0
  return sum/n


def average_out_of_transit(time,flux,duration,period,t0):
  n=0
  sum=0
  for i in range(0,len(time)):
    t=(time[i]-t0)%period
    assert 0<=t and t<=period
    if (duration/2<=t and t<=period-duration/2) and time[i]>0.1:
      n+=1
      sum+=flux[i]
  if n==0:
    return 0
  return sum/n


def convert(row_id,tic_id,Sectors,P,pphase1,pphase2,writer):
    dirlist=[filename(Sectors,tic_id)]
    lc,tu,n,ntrue,nq,inval, bjdoffset,t0arr, t, det, info=tes.load_tesslc(dirlist)
    Epoc=(pphase1+pphase2)/2 #+(t0arr[0])
    Period=P
    Duration=(pphase2-pphase1)*24
    if average_in_transit(tu,lc,Duration/24,Period,Epoc)<1:
        return False
    if tic_id==30407196:
        print(t0arr[0],average_in_transit(tu,lc,Duration/24,Period,Epoc),average_out_of_transit(tu,lc,Duration/24,Period,Epoc))
    Epoc+=t0arr[0]
    writer.writerow([row_id,tic_id,'J',0,0,Epoc,Period,Duration,0,Sectors,0,0])
    return True


if __name__ == '__main__':
    print('start process')
    sys.path.append('D:\\SynologyDrive\\Univ\\kenkyuu\\gtrap\\gtrap')
    args=sys.argv
    tic_csv_name=args[1]
    print('reading ' +tic_csv_name)
    output_csv_name=args[2]
    print('output will be written to '+output_csv_name)
    output_csv=open(output_csv_name,'w',newline="")
    writer=csv.writer(output_csv)
    writer.writerow(['row_id','tic_id','Disposition','RA','Dec','Epoc','Period','Duration','Transit_Depth','Sectors','SN','Qingress'])
    rowid=0
    with open(tic_csv_name) as f:
        reader=csv.reader(f)
        l=[row for row in reader]
        numtic=len(l)-1
        print('process '+str(numtic)+' stars')
        for i in range(1,len(l)):
            print('processing '+str(i)+'/'+str(numtic))
            res=convert(row_id=rowid,tic_id=int(l[i][0]),Sectors=int(l[i][1]),P=float(l[i][2]),pphase1=float(l[i][6]),pphase2=float(l[i][7]),writer=writer)
            if res:
              rowid+=1
    output_csv.close()
