import sys
import glob
import os
import csv
from astroquery.mast import Catalogs
import numpy as np

def star_query(tic):
    target_name="TIC "+str(tic)
    search_radius_deg=0.002
    catalogTIC=Catalogs.query_object(target_name,radius=search_radius_deg,catalog="TIC")
    #where_closest = np.argmin(catalogTIC['dstArcSec'])
    where_closest=-1
    for i in range(len(catalogTIC)):
        if str(catalogTIC['ID'][i])==str(tic):
            where_closest=i
    print(tic,catalogTIC['ID'][where_closest],len(catalogTIC))
    assert where_closest != -1

    starparam = {}
    starparam["mass"] = catalogTIC["mass"][where_closest]
    starparam["rad"] = catalogTIC["rad"][where_closest]
    starparam["Tmag"] = catalogTIC["Tmag"][where_closest]
    starparam["logg"] = catalogTIC["logg"][where_closest]
    starparam["Teff"] = catalogTIC["Teff"][where_closest]
    return starparam


args=sys.argv
dir=sys.argv[1]
files=glob.glob(os.path.join(dir,'*.png'))
files2=glob.glob('D:\\SynologyDrive\\Univ\\kenkyuu\\Astronet-Triage\\astronet\\data\\prediction\\*_scores.csv')
bls={}
for file in files2:
    with open(file) as f:
        reader=csv.reader(f)
        l=[row for row in reader]
        for i in range(1,len(l)):
            bls[(int(l[i][0]),int(l[i][5]))]=(float(l[i][1]),float(l[i][2]),float(l[i][3]),float(l[i][4]))#score,epoc,period,duration

output_csv_name=args[2]
print('output will be written to '+output_csv_name)
output_csv=open(output_csv_name,'w',newline="")
writer=csv.writer(output_csv)
writer.writerow(['tic_id','mass','rad','Tmag','logg','Teff','Epoc','Period','Duration','sector','score'])
for file in files:
    filename=os.path.splitext(os.path.basename(file))[0]
    sector,tic=filename.split('_')
    sector=int(sector[2:])
    tic=int(tic)
    score,epoc,period,duration=bls[(tic,sector)]
    param=star_query(tic)
    writer.writerow([tic,param['mass'],param['rad'],param['Tmag'],param['logg'],param['Teff'],epoc,period,duration,sector,score])
