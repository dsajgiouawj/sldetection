# Copyright 2018 The TensorFlow Authors.
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

"""Functions for reading and preprocessing light curves."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from light_curve_util import tess_io
from light_curve_util import median_filter
from light_curve_util import util
from statsmodels.robust import scale


# use this to trim multi-sector light curves to just the latest sector
sector_start = {2: 1354.10475587519, 3: 1381.70892156158, 4: 1410.91724195171, 5: 1437.97973532546}


class EmptyLightCurveError(Exception):
    """Indicates light curve with no points in chosen time range."""
    pass


def read_and_process_light_curve(tic, sector=1, injected=False, inject_dir='/pdo/users/yuliang', is_multi=False):
  """Reads an already detrended light curve.

  Args:
    tic: TIC id of the target star.

  Returns:
    time: 1D NumPy array; the time values of the light curve.
    flux: 1D NumPy array; the normalized flux values of the light curve.

  Raises:
    IOError: If the light curve files for this TIC ID cannot be found.
    EmptyLightCurveError: If light curve has no points in given time range.
  """
  # Read the TESS light curve.
  file_names = tess_io.tess_filenames(tic, sector=sector, injected=injected, inject_dir=inject_dir)
  if not file_names:
    tf.logging.info("Failed to find light curve files in for TIC ID %s" % (tic))
    raise IOError

  all_time, all_mag = tess_io.read_tess_light_curve(file_names)

  if len(all_time) < 1:
      tf.logging.info("Empty light curve. Skipped TIC id %s" % (tic))
      raise EmptyLightCurveError

  #mad = scale.mad(all_mag)
  #valid_indices = np.where(all_mag > np.median(all_mag)-5*mad)
  #all_mag = all_mag[valid_indices]
  #all_time = all_time[valid_indices]
  #all_flux = 10.**(-(all_mag - np.median(all_mag))/2.5)
  #all_flux=all_mag-np.median(all_mag)
  all_flux=all_mag

  if sector > 1 and is_multi:
      current = np.where(all_time >= sector_start[sector])
      all_time = all_time[current]
      all_flux = all_flux[current]
  return all_time, all_flux


def invert_out_of_transit(time,flux,duration,period,t0):
  for i in range(0,len(time)):
    t=(time[i]-t0)%period
    assert 0<=t and t<=period
    if duration/2<=t and t<=period-duration/2:
      flux[i]=2-flux[i]
  return time,flux


def phase_fold_and_sort_light_curve(time, flux, period, t0):
  """Phase folds a light curve and sorts by ascending time.

  Args:
    time: 1D NumPy array of time values.
    flux: 1D NumPy array of flux values.
    period: A positive real scalar; the period to fold over.
    t0: The center of the resulting folded vector; this value is mapped to 0.

  Returns:
    folded_time: 1D NumPy array of phase folded time values in
        [-period / 2, period / 2), where 0 corresponds to t0 in the original
        time array. Values are sorted in ascending order.
    folded_flux: 1D NumPy array. Values are the same as the original input
        array, but sorted by folded_time.
  """
  # Phase fold time.
  time = util.phase_fold_time(time, period, t0)

  # Sort by ascending time.
  sorted_i = np.argsort(time)
  time = time[sorted_i]
  flux = flux[sorted_i]

  return time, flux


def generate_view(time, flux, num_bins, bin_width, t_min, t_max,
                  normalize=True):
  """Generates a view of a phase-folded light curve using a median filter.

  Args:
    time: 1D array of time values, phase folded and sorted in ascending order.
    flux: 1D array of flux values.
    num_bins: The number of intervals to divide the time axis into.
    bin_width: The width of each bin on the time axis.
    t_min: The inclusive leftmost value to consider on the time axis.
    t_max: The exclusive rightmost value to consider on the time axis.
    normalize: Whether to center the median at 0 and minimum value at -1.

  Returns:
    1D NumPy array of size num_bins containing the median flux values of
    uniformly spaced bins on the phase-folded time axis.
  """
  view = median_filter.median_filter(time, flux, num_bins, bin_width, t_min,
                                     t_max)
  if normalize:
    view -= np.median(view)
    view /= np.abs(np.min(view))  # In pathological cases, min(view) is zero...

  return view


def global_view(time, flux, period, num_bins=201, bin_width_factor=1.2/201):
  """Generates a 'global view' of a phase folded light curve.

  See Section 3.3 of Shallue & Vanderburg, 2018, The Astronomical Journal.
  http://iopscience.iop.org/article/10.3847/1538-3881/aa9e09/meta

  Args:
    time: 1D array of time values, sorted in ascending order.
    flux: 1D array of flux values.
    period: The period of the event (in days).
    num_bins: The number of intervals to divide the time axis into.
    bin_width_factor: Width of the bins, as a fraction of period.

  Returns:
    1D NumPy array of size num_bins containing the median flux values of
    uniformly spaced bins on the phase-folded time axis.
  """
  return generate_view(
      time,
      flux,
      num_bins=num_bins,
      bin_width=period * bin_width_factor,
      t_min=-period / 2,
      t_max=period / 2)


def twice_global_view(time, flux, period, num_bins=402, bin_width_factor=1.2 / 402):
  """Generates a 'global view' of a phase folded light curve at 2x the BLS period.

  See Section 3.3 of Shallue & Vanderburg, 2018, The Astronomical Journal.
  http://iopscience.iop.org/article/10.3847/1538-3881/aa9e09/meta
  If single transit, this is pretty much identical to global_view.

  Args:
    time: 1D array of time values, sorted in ascending order, phase-folded at 2x period.
    flux: 1D array of flux values.
    period: The period of the event (in days).
    num_bins: The number of intervals to divide the time axis into.
    bin_width_factor: Width of the bins, as a fraction of period.

  Returns:
    1D NumPy array of size num_bins containing the median flux values of
    uniformly spaced bins on the phase-folded time axis.
  """
  return generate_view(
      time,
      flux,
      num_bins=num_bins,
      bin_width=period * bin_width_factor,
      t_min=-period,
      t_max=period)


def local_view(time,
               flux,
               period,
               duration,
               num_bins=61,
               bin_width_factor=0.16,
               num_durations=2):
  """Generates a 'local view' of a phase folded light curve.
  See Section 3.3 of Shallue & Vanderburg, 2018, The Astronomical Journal.
  http://iopscience.iop.org/article/10.3847/1538-3881/aa9e09/meta
  Args:
    time: 1D array of time values, sorted in ascending order.
    flux: 1D array of flux values.
    period: The period of the event (in days).
    duration: The duration of the event (in days).
    num_bins: The number of intervals to divide the time axis into.
    bin_width_factor: Width of the bins, as a fraction of duration.
    num_durations: The number of durations to consider on either side of 0 (the
        event is assumed to be centered at 0).
  Returns:
    1D NumPy array of size num_bins containing the median flux values of
    uniformly spaced bins on the phase-folded time axis.
  """
  return generate_view(
      time,
      flux,
      num_bins=num_bins,
      bin_width=duration * bin_width_factor,
      t_min=max(-period / 2, -duration * num_durations),
      t_max=min(period / 2, duration * num_durations))


def mask_transit(time, duration, period, mask_width=2, phase_limit=0.1):
    """

    :param time: 1D array of time values, folded and sorted in ascending order, with the transit located at time 0.
    :param duration: The duration of the event (in days).
    :param period: the period of the event (in days).
    :param mask_width: number of durations to mask out.
    :param phase_limit: minimum phase to search for secondary eclipse.
    :return: mask: 1D array of booleans
    """
    mask = [(abs(t) > duration*mask_width/2) and (abs(t) > period*phase_limit) for t in time]
    return np.array(mask)
