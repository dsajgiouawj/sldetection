import argparse
import shutil
import pandas as pd

parser = argparse.ArgumentParser(description='cp')
parser.add_argument('-l', nargs=1, help='tic,scc list', type=str)
args = parser.parse_args()

if args.l:
    tslist=pd.read_csv(args.l[0],delimiter=",")
    ticlist=tslist["TIC"]
    scclist=tslist["SCC"]

filelist=[]
for ii,tic in enumerate(ticlist):
    scc=scclist[ii]
    sector = int(scc.split("_")[0])
    if sector<11:
        filelist.append("/manta/pipeline/CTL2/tess_"+str(tic)+"_"+str(scc)+".h5")
    else:
        filelist.append("/stingray/pipeline/CTL2/tess_"+str(tic)+"_"+str(scc)+".h5")
        

for f in filelist:
    shutil.copy2(f,"./")
        
