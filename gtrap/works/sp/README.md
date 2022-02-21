
# NN
How to use astronet...
/bin/bash
conda activate tf-gpu


# TESS (SLC)

```
gtls_slctess.py -p (pick up)
gtls_slctess.py -q (no injection)
gtls_slctess.py (injection)
```


## When updating the LCs

- 0. Update the data list 

data/python_updatelist/make_list_sector.py
list are in data/ctl.list

- 1. Generate training/test datasets

Use batch/makemocksh.py and examples/sh/*.sh


- 2. Train the DNN


- 3. Pick up the pulse candidates from all h5.

Use batch/makepicksh.py.

- 4. Do DNN!


## Updating training sets

train/train? is trained data sets/


-tips
use cph5fromlist.py and .list to collect h5 for transfer. 

# Kepler STE

- gtls_mockkepler.py -- generate mock template after the TLS identifier
 use genGroundTrues.sh for sequential use.
- astronet.py -- a KERAS version of astronet (training)
- gtls_pickkepler.py pickup by TLS for all kepler LCs
- astronet_picklc.py prediction for all kepler LCs


## Tuning of TLS

test_pick_kelp.sh : test of TLS algorithm using the clean sample of the KeLP catalog.
test_pick_kelp.py : comparison

