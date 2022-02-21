import sys
import os
import time
import matplotlib.pyplot as plt
import math
import numpy as np
import csv
import shutil

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


def bls(sector,tic,output_directory):
    fn=filename(sector,tic)
    shutil.copy(fn,output_directory)


if __name__ == '__main__':
    print('start process')
    args=sys.argv
    sector=int(args[1])
    print('input sector is '+str(sector))
    tic_csv_name=args[2]
    print('reading ' +tic_csv_name)
    output_directory=args[3]
    print('output will be written to '+output_directory)
    with open(tic_csv_name) as f:
        reader=csv.reader(f)
        l=[row for row in reader]
        numtic=len(l)-1
        print('process '+str(numtic)+' stars')
        for i in range(800001,1000001):
            print('processing '+str(i)+'/'+str(numtic))
            bls(sector,int(l[i][0]),output_directory)
