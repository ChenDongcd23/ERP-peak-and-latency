import numpy as np
import pandas as pd
from mne.utils import _time_mask

import mne



'''
This script measures ERP values using an algorithm similar to ERPLAB, 
see: https://erpinfo.org/erplab

The current version supports only 
peak amplitude, peak latency, and fractional peak latency

TODO: mean amplitude, area amplitude, and fractional area latency.
'''



# =========================================================
# Main Function
# =========================================================
def find_local_peak(
    evoked,
    picks=None,
    neighborhood=1,
    peak_polarity="positive",
    measure="fraclat",
    fraction=None,
    frac_direction="left",
    peak_replace="abs",
    meas_win=None,
    frac_win_mode="on",
    average=False,
    interp=True
):
    """
    Detect local peaks in an Evoked object.

    Parameters
    ----------
    evoked : mne.Evoked

    picks : str | array_like | None
        Channels to include. If None, all channels are used.
    neighborhood : int
        Neighborhood size for local mean comparison.
    peak_polarity : {"positive", "negative"}
        Polarity of the peak to detect.
    measure : {"peaklat", "fraclat", "both"}
        Type of latency measure to return.
    fraction : float | None
        Fraction (0 < fraction < 1) of peak amplitude for fractional latency.
    frac_direction : {"left", "right"}
        Direction to search for fractional crossing.
    peak_replace : {"abs", "off"}
        Fallback strategy if no local peak is found.
    meas_win : tuple | None
        Time window (tmin, tmax) for peak detection.
    frac_win_mode : {"on", "off"}
        Whether fractional latency is computed within the measurement window.
    average : bool
        If True, average across selected channels before detection.
    interp : bool
        If True, use linear interpolation for fractional latency.

    Returns
    -------
    pandas.DataFrame
        DataFrame indexed by channel with peak values and latencies.
    """
    
    ## 0. Evoked object check
    if not hasattr(evoked, "info") or not hasattr(evoked, "_data"):
        raise TypeError("`evoked` must be an Evoked object with 'info' and '_data' attributes.")

    ## 1. Preparation
    sfreq = evoked.info["sfreq"]
    data, times, ch_names = _prepare_data(evoked, picks, average)
    n = len(times)
    data *= 1e6

    ## 2. Basic validation
    if n < 3:
        raise ValueError("Signal must have at least 3 points.")
    if not np.all(np.diff(times) > 0):
        raise ValueError("`times` must be strictly increasing.")
    if neighborhood < 1 or neighborhood >= n // 2:
        raise ValueError("`neighborhood` must be >=1 and < n/2")
    if peak_polarity not in ["positive", "negative"]:
        raise ValueError("Invalid peak_polarity, only 'postive' or 'negative'.")
    if measure not in ["peaklat", "fraclat", "both"]:
        raise ValueError("Invalid measure, please choose in 'peaklat', 'fraclat', 'both'.")
    # meas_win must be a 2-element tuple, list or array. 

    # mode choose
    if measure in ["fraclat", "both"]:
        if fraction is None:
            compute_frac = False
        else:
            if not (0 < fraction < 1):
                raise ValueError("`fraction` must be in (0,1).")
            compute_frac = True
    else:
        compute_frac = False

    ## 3. Main loop
    rows = []
    for idx, ch_name in enumerate(ch_names):
        datum = data[idx]

        # measure window
        if meas_win is not None:
            mask_win = _time_mask(times, tmin=meas_win[0], tmax=meas_win[1], sfreq=sfreq, include_tmax=True)
        else:
            mask_win = np.ones_like(times, dtype=bool)

        # find peak
        v_local, lat_local, pos_local = _detect_local_peak_in_window(
            datum, times, mask_win, neighborhood, peak_polarity, peak_replace
        )

        # fractional peak latency
        lat_frac = np.nan
        if compute_frac:
            if frac_win_mode == "off":
                data_frac = datum
                times_frac = times
                pos_start = pos_local
            elif frac_win_mode == "on":
                data_frac = datum[mask_win]
                times_frac = times[mask_win]
                if pos_local is None:
                    pos_start = None
                else:
                    orig_idx = pos_local
                    window_indices = np.where(mask_win)[0]
                    if orig_idx in window_indices:
                        pos_start = np.where(window_indices == orig_idx)[0][0]
                    else:
                        data_frac = datum
                        times_frac = times
                        pos_start = pos_local
            else:
                raise ValueError("frac_win_mode must be 'on' or 'off'")

            lat_frac = _compute_fractional_latency(
                data_frac, times_frac, v_local, pos_start, 
                fraction, frac_direction, interp
            )

        # output
        if measure == "peaklat":
            rows.append({"channel": ch_name, "value": v_local, "latency": lat_local})
        elif measure == "fraclat":
            rows.append({"channel": ch_name, "value": v_local, "frac_latency": lat_frac})
        else:
            rows.append({"channel": ch_name, "value": v_local, "latency": lat_local, "frac_latency": lat_frac})

    results_df = pd.DataFrame(rows).set_index("channel")
    return results_df


# =========================================================
def _prepare_data(evoked, picks=None, average=False):
    """
    Extract data, times, and channel names from the Evoked object.
    Optionally average across selected channels.
    """
        
    datx = evoked.get_data(picks=picks)
    timx = evoked.times

    if average:
        datx = np.mean(datx, axis=0, keepdims=True)
        ch_names = ["Ave"]
    else:
        if picks is None:
            ch_names = evoked.ch_names
        else:
            ch_names = [evoked.ch_names[p] if isinstance(p, int) else p for p in picks]

    return datx, timx, ch_names


# =========================================================
def _detect_local_peak_in_window(datax, times, mask_win, neighborhood, peak_polarity, peak_replace):
    """
    Detect a local peak within a masked time window.
    Falls back to absolute peak if no local candidate is found.
    """
    
    pol = 1 if peak_polarity == "positive" else -1

    # set mask
    data_win = datax[mask_win]
    times_win = times[mask_win]
    idx_map = np.where(mask_win)[0]
    n = len(data_win)

    # absolute peak
    v_abs = data_win[np.argmax(data_win)] if pol == 1 else data_win[np.argmin(data_win)]
    lat_abs = times_win[np.argmax(data_win)] if pol == 1 else times_win[np.argmin(data_win)]
    pos_abs = idx_map[np.argmax(data_win)] if pol == 1 else idx_map[np.argmin(data_win)]

    # local peak
    # adjacent comparison by vector
    left = data_win[1:-1] > data_win[:-2] if pol == 1 else data_win[1:-1] < data_win[:-2]
    right = data_win[1:-1] > data_win[2:] if pol == 1 else data_win[1:-1] < data_win[2:]
    mask_adj = left & right
    mask = np.zeros_like(mask_adj, dtype=bool)

    # neighborhood mean comparison
    if neighborhood > 1 and n > 2 * neighborhood:
        kernel = np.ones(neighborhood) / neighborhood
        res = np.convolve(data_win, kernel, mode="valid") #mean series
        
        mid_section = data_win[neighborhood : -neighborhood] #comparable series
        left_mean = res[:len(mid_section)] #left_mean
        right_mean = res[neighborhood + 1:] #right_mean:
        
        if pol == 1:
            mask_mean = (mid_section > left_mean) & (mid_section > right_mean)
        else:
            mask_mean = (mid_section < left_mean) & (mid_section < right_mean)
        
        mask_start = neighborhood - 1
        mask_end = mask_start + len(mask_mean)
        mask[mask_start : mask_end] = mask_mean
        
        mask &= mask_adj
    else:
        mask = mask_adj

    candidate_idx = np.where(mask)[0] + 1

    if len(candidate_idx) > 0:
        local_idx = candidate_idx[np.argmax(data_win[candidate_idx])] if pol == 1 else candidate_idx[np.argmin(data_win[candidate_idx])]
        pos_local = idx_map[local_idx]
        return datax[pos_local], times[pos_local], pos_local

    # fallback
    if peak_replace == "abs":
        return v_abs, lat_abs, pos_abs

    return np.nan, np.nan, None


# =========================================================
def _compute_fractional_latency(datax, times, v_local, pos_start, fraction, frac_direction, interp=False):
    """
    Compute fractional peak latency by searching from the peak
    toward the specified direction until the signal crosses
    a fraction of the peak amplitude. (Interpolation is an optional)
    """
    
    if pos_start is None:
        return np.nan

    target = v_local * fraction
    step = -1 if frac_direction == "left" else 1
    n = len(datax)

    idx = pos_start + step 
    if idx < 0 or idx >= n: 
        return np.nan 
    
    prev_idx = pos_start 
    prev_val = datax[pos_start] 
    
    while 0 <= idx < n: 
        val = datax[idx] 
        crossed = (prev_val - target) * (val - target) <= 0 
        
        if crossed and prev_val != val: 
            if interp: 
                t0, t1 = times[prev_idx], times[idx] 
                v0, v1 = prev_val, val 
                t_frac = t0 + (target - v0) / (v1 - v0) * (t1 - t0) 
                return t_frac 
            else: 
                if abs(prev_val - target) <= abs(val - target):
                    return times[prev_idx]
                else:
                    return times[idx] 
            
        prev_idx = idx 
        prev_val = val 
        idx += step

    return np.nan


# ===================================================================

# Test 
path = r'D:/test_evoked_ave.fif'
test_evoked = mne.read_evokeds(path, condition=0)

r = find_local_peak(test_evoked,
                    picks=None,
                    neighborhood=1,
                    peak_polarity="negative",
                    measure="both",
                    fraction=0.5,
                    frac_direction="left",
                    peak_replace="abs",
                    meas_win=(-0.2, 0),
                    frac_win_mode="off",
                    average=True,
                    interp=False)

print(r)

#             value  latency  frac_latency
# channel

# Ave     -3.761501   -0.136        -0.208
