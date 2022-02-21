from PIL import Image
import os
import sys

im=Image.open(sys.argv[1])
im.crop((1300,0,2250,1000)).save(sys.argv[1][0:-4]+'_ph.png',quality=95)
