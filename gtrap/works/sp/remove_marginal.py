import numpy
import pandas as pd
import shutil
import glob
dat=pd.read_csv("marginal_train0.txt")
ticlist=dat["TIC"]
for tic in ticlist:
    print(tic)
    pathname='/home/kawahara/gtrap/examples/mocklcslc/*/npz/*'+str(tic)+'*.npz'
    dst_path="/home/kawahara/gtrap/examples/mocklcslc/marginal"
    for p in glob.glob(pathname):
        shutil.move(p, dst_path)
