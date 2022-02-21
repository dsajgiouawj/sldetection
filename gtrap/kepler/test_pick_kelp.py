import pandas as pd

dat1=pd.read_csv("kelp_checkoutput.100v2.txt")
dat0=pd.read_csv("kelp_checkoutput.100v3_n10.txt")
dat=pd.read_csv("kelp_checkoutput.100v4.txt")



w1=dat1["N"]
w0=dat0["N"]
w=dat["N"]

print("v2 (n=3)",len(w1[w1>0]),"/",len(w1))
print("v3 (n=10)",len(w0[w0>0]),"/",len(w0))
print("v4 (n=10) & new criterion",len(w[w>0]),"/",len(w))

