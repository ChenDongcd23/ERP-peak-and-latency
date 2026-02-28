import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mne

from get_peak import * 
from pandasgui import show


## 0.Read data =======================================================

path = r'D:/test_evoked_ave.fif'
test_evoked = mne.read_evokeds(path, condition=0)

path2 = r'D:/test_evoked2_ave.fif'
test_evoked2 = mne.read_evokeds(path2, condition=0)


## 1. Test function ===================================================

# peak amplitude and latency ------------------------------------------
peak_r = find_local_peak(
    test_evoked,
    picks = None,
    neighborhood = 1,
    peak_polarity = "negative",
    measure = "both",
    fraction = 0.5,
    frac_direction = "left",
    peak_replace = "abs",
    meas_win = (-0.2, 0),
    frac_win_mode = "off",
    average = True,
    interp = False)

print(peak_r)


# area amplitude and frac_lat------------------------------------------
area_r = get_area(
    test_evoked,
    meas_win = (-0.2, 0), 
    picks =None,
    mode = 'neg',
    boundary_mode = 'fixed',
    coi = 2,
    average = True,
    frac = 0.5,
    side = 'left',
    boundary_log = False
)

print(area_r)

# auto boundary detection 
area_b, log = get_area(
    test_evoked,
    meas_win = (-0.2, 0), 
    picks =None,
    mode = 'pos',
    boundary_mode = 'auto',
    coi = 0,
    average = True,
    frac = 0.5,
    side = 'left',
    boundary_log = True
)

print(area_b)
print(log)
show(log)

## 2. Result ==========================================================

# result 1
#   channel     value     latency     frac_latency
#   Ave         -3.762    -0.136      -0.204

# result 2
#   channel     area      mean_amp    frac_area_lat
#   Ave         0.503     2.517       -0.120

# result 3
#   orig_start     orig_end     new_start     new_end
#   -0.2           0.0	        -0.344	      0.040
