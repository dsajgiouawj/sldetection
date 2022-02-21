# Copyright 2018 Liang Yu.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Functions for reading TESS data."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path
#import h5py
from astropy.io import fits
import numpy as np

from tensorflow import gfile
import sys


def tess_filenames(tic,
                     base_dir='', # not used
                     sector=1,
                     injected=False, # not used
                     inject_dir='/pdo/users/yuliang', # not used
                     check_existence=True):
    """Returns the light curve filename for a TESS target star.

    Args:
      tic: TIC of the target star. May be an int or a possibly zero-
          padded string.
      base_dir: Base directory containing Kepler data.
      sector: Int, sector number of data.
      cam: Int, camera number of data.
      ccd: Int, CCD number of data.
      injected: Bool, whether target also has a light curve with injected planets.
      injected_dir: Directory containing light curves with injected transits.
      check_existence: If True, only return filenames corresponding to files that
          exist.

    Returns:
      filename for given TIC.
    """
    tic = str(tic).zfill(16)
    if sector <= 13:
        base_dir = r'F:\QLP\Year1\s'
    elif sector <=26:
        base_dir = 'G:\QLP\Year2\s'
    else:
        raise ValueError("sector>=27 not supported now")
    sector = str(sector).zfill(4)
    base_dir = base_dir + sector + '\\' + tic[0:4] + '\\' + tic[4:8] +'\\' + tic[8:12] + '\\' + tic[12:16]

    if not injected:
        # modify this as needed
        base_name = "hlsp_qlp_tess_ffi_s" + sector + '-' + tic + '_tess_v01_llc.fits'
        filename = os.path.join(base_dir, base_name)
    else:
        raise ValueError("injected=True not supported now")
        # filename = os.path.join(inject_dir, tic + '.lc')

    if not check_existence or gfile.Exists(filename):
        return filename
    return


def read_tess_light_curve(filename, flux_key='Normalized SAP_FLUX', invert=False):
    """Reads time and flux measurements for a Kepler target star.

    Args:
      filename: str name of h5 file containing light curve.
      flux_key: Key of h5 column containing detrended flux.
      invert: Whether to invert the flux measurements by multiplying by -1.

    Returns:
      time: Numpy array; the time values of the light curve.
      flux: Numpy array corresponding to the time array.
    """
    with fits.open(open(filename, "rb")) as hdu_list:
        light_curve = hdu_list["LIGHTCURVE"].data
        time = light_curve.TIME
        flux = light_curve.KSPSAP_FLUX

    # time = np.array(f["LightCurve"]["BJD"])
    # mag = np.array(api[flux_key])
    quality_flag = np.where(light_curve.QUALITY == 0)
    # Remove outliers
    time = time[quality_flag]
    flux = flux[quality_flag]

    # Remove NaN flux values.
    valid_indices = np.where(np.isfinite(flux))
    time = time[valid_indices]
    flux = flux[valid_indices]

    if invert:
        flux *= -1

    return time, flux
