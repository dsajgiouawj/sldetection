# GTRAP: Gpu-based TRAnsiting Planet candidate detector

GPL licence

```
setenv CPLUS_INCLUDE_PATH /install_directory/gtrap/include
setenv PYTHONPATH /install_directory/gtrap
```

pip install mysql-connector-python-rf


# examples

- gtls_kepler: GPU Trapezoid Least Square (GTLS) for the Kepler data. Used in [Kawahara and Masuda (2019)](http://arxiv.org/abs/1904.04980)

For the data, see [here](http://secondearths.sakura.ne.jp/gtrap/).


## dnn

CNN classifier using Keras. From astronet.
Set your Keras environment.


# gtrap

## gtls

GPU-based TLS (Trapezoid Least Sqaure).

- gtls: simple gpu-based TLS

## geebls

GPU-based BLS, pycuda-version of eebls.f (Kovac+2002). The codes use single precision float array. This requirement makes slight difference in the result between eebls.f and this code. Shared memory is used to cache folded light curve and temporary Signal Residue (SR). By default, GBLS allows a batch computation. An input array should be an image of light curves, which contains a set of N-light curves. 

- geebls_simple: simple gpu-based BLS
- geebls: gpu-based BLS with mask, offset, and a non-common time array, phase information

mask: Masked bins are ignored for the BLS computation. Values in a time array should be negative for the masked indices.
offset: This allows an extrenal input of the offset of the lightcurve.
non-common time array: This allows a different time sequence for each light curve. Set the time array which has the same dimension to that of the lightcurve image array.

Smoothing of light curves before the BLS generally increases the BLS signal. 
scipy.signal.medfilt provides a good detrending method.

### gfilter

GBLS also has a gpu-based smoother, which devide a light curve by the median filtered curve.

- gfilter: a gpu-based smoother

The algorithm for the fast median filter is based on the chapter of "Implementing a fast median filter" by Gilles Perrot from the textbook, "Designing Scientific Applications on GPUs".

<<<<<<< HEAD
## genmonck

Generating LCs with mock transit (for training).

## picktrap

## gnet

CNN classifier using Keras.
=======
>>>>>>> 057afdf07787aeee485051d82cdc7c7279cef019

#external modules used in gtrap

- gpytorch: for the fast GP
