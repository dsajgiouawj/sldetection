import pandas as pd
import h5py
import numpy as np
import mysql.connector
from urllib.parse import urlparse
import vineyard.modules.io as vmi

p_work = {
    "data_type" : "CTL",
       "sector" : [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
       "camera" : [1, 2, 3, 4],
         "chip" : [1, 2, 3, 4],
         "jobs" : 30,
}

rad=[]
mass=[]
files=[]
#mysql -h 133.11.231.118 -u {user} -p TESS
#url = urlparse('mysql://fisher:atlantic@133.11.231.118:3306/TESS')
url = urlparse('mysql://fisher:atlantic@133.11.229.168:3306/TESS')
conn = mysql.connector.connect(
        host = url.hostname or '133.11.229.168',
        port = url.port or 3306,
        user = url.username or 'fisher',
        password = url.password or 'atlantic',
        database = url.path[1:],
)
cur = conn.cursor()



for sector in p_work["sector"]:
    for camera in p_work["camera"]:
        for chip in p_work["camera"]:


#for sector in [1]:
#    for camera in [1]:
#        for chip in [1]:        
            h5list = vmi.glob_h5(p_work["data_type"], sector, camera, chip, 1)
            j=0
            nlen=len(h5list)
            tag=str(sector)+"_"+str(camera)+"_"+str(chip)
#            print(h5list)
            for fn in h5list:
                files.append(fn)
                j=j+1
                if(np.mod(j,10000)==0):
                    print(j,"/",nlen)
                tic=fn.split("_")[1]
                try:
            #        cur.execute('SELECT rad,mass FROM CTLv7 where ID='+str(tic))
                    cur.execute('SELECT rad,mass FROM CTLchip'+tag+' where ID='+str(tic))
                    out=cur.fetchall()[0]
                    out=np.array(out) #rad, mass
                    rad.append(out[0])
                    mass.append(out[1]) 
#                    print(rad,mass)
                except:
                    rad.append(-1.0)
                    mass.append(-1.0)
rad=np.array(rad)
mass=np.array(mass)
    
np.savez("ctl.list.npz",files,rad,mass)
