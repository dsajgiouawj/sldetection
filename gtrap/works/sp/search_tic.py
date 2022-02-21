import numpy as np
dat=np.load("../data/step3.list.npz")
files=dat["arr_0"]

for i in range(0,200):
    print(files[i],i)
