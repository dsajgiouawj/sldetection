import time
import matplotlib.pyplot as plt
import math
import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda
import pycuda.compiler
from pycuda.compiler import SourceModule

def gbls_module ():
    source_module = SourceModule("""

    #include <stdio.h>
    #define MINBIN 5
    #define INVT 0.0
    extern __shared__ float cache[]; 

    #include "geebls.h"

    """,options=['-use_fast_math',r'-ID:\SynologyDrive\Univ\kenkyuu\gtrap\include'])

    return source_module

if __name__ == "__main__":

    print ("gpu eebls")
