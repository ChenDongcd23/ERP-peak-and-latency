import numpy as np
import pandas as pd
from mne.utils import _time_mask, _check_option
from scipy.interpolate import interp1d
from scipy import integrate


'''
This script measures ERP values using an algorithm similar to ERPLAB, 
see: https://erpinfo.org/erplab

The current version supports  
peak amplitude, peak latency, and fractional peak latency
area amplitude, mean amplitude, and fractional area latency

TODO: Further test function `_auto_boundary`, specially the auto boundary
detection using `coi`.
'''


# Peak ERP =========================================================
# Main Function
# ==================================================================
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
        
    datx = evoked.get_data(picks=picks, units=dict(eeg='uV'))
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



# Area ERP =========================================================
# Main Function
# ==================================================================

# TODO: 
# 1. get the boudary of time window based on ROI waveform
# 2. use the boudary back to single channels

# Area 
def get_area(
    evoked,
    meas_win, 
    picks='None',
    mode='abs',
    boundary_mode='auto',
    coi=0,
    average=False,
    frac = 0.5,
    side = 'left',
    boundary_log = False
):
    '''
    Calculate the area under the curve and fractional area latency for ERP data.

    This function identifies a specific segment of the signal (based on time windows 
    and auto boundary detection) and computes its area, mean amplitude, and the time 
    point where a certain percentage of the area is reached.

    Parameters:
    -----------
    evoked : mne.Evoked
        The evoked data object.
    meas_win : list | tuple
        The time window [start, end] in seconds for measurement.
    picks : str | list | None
        Channels to include. Defaults to 'None' (all).
    mode : str
        Method to process the signal: 
        'abs' (absolute), 'neg' (negative only), 'pos' (positive only), 'intg' (original).
    boundary_mode : str
        How to define the integration edges:
        'fixed': Use `meas_win` strictly.
        'auto': Automatically find zero-crossings around the peak.
        'hybrid': Use the intersection of fixed and auto boundaries.
    coi : int
        Condition of interest for boundary refinement (used in _auto_boundary).
    average : bool
        If True, average across selected channels before calculation.
    frac : float
        The fraction of the area (0 to 1) to calculate latency for (e.g., 0.5 for median).
    side : str
        Direction to accumulate the area: 'left' (start to end) or 'right' (end to start).

    Returns:
    --------
    pd.DataFrame
        A DataFrame containing channel names, area, mean amplitude, and fractional latency.
    '''
    
    _check_option("mode", mode, ["abs", "neg", "pos", "intg"])
    _check_option("boundary_mode", boundary_mode, ["fixed", "auto", "hybrid"])
    if not (0 < frac < 1):
        raise ValueError("`fraction` must be in (0,1).")
    
    # basic data
    datx, timx, ch_names = _prepare_data(evoked, picks=picks, average=average)
    
    #trans time to sample
    twin = [
        int(np.argmin(np.abs(timx - meas_win[0]))),
        int(np.argmin(np.abs(timx - meas_win[1])))
    ]
    
    area_results = []
    mean_amps = []
    lat_results = []
    b_log = []
    for i, ch_data in enumerate(datx):
        if boundary_mode == 'fixed':
            a, b = twin
        else:
            a_auto, b_auto = _auto_boundary(ch_data, twin, coi)
            
            if boundary_mode == "auto":
                a, b = a_auto, b_auto
            elif boundary_mode == "hybrid":
                a = max(a_auto, twin[0])
                b = min(b_auto, twin[1])
                
            # boundary log
            if boundary_log:
                log = _boundary_report(ch_names[i], twin, (a,b), timx)
                if log:
                    b_log.append(log)
                
        # area calculator
        seg = ch_data[a:b+1]
        tseg = timx[a:b+1]
        area, mean_amp = _area_calculation(seg, tseg, mode)
        area_results.append(area)
        mean_amps.append(mean_amp)
        
        # find frac_area_lat 
        seg_t = _apply_mode(seg, mode)
        frac_area_lat = _frac_area_latency(seg_t, tseg, area, fraction=frac, side=side)
        lat_results.append(frac_area_lat)
        
    df_res =  pd.DataFrame({
        "channel": ch_names,
        "area": area_results,
        "mean_amp": mean_amps,
        "frac_area_lat": lat_results
    })
    
    if boundary_log:
        df_boundary = pd.DataFrame(b_log)
        return df_res, df_boundary
    else:
        return df_res
    

# auto boundary mode
def _auto_boundary(data, latsam, coi=0):
    
    '''
    Automatically detects boundaries for a wave component.
    
    Inspired by the ERPlab Toolbox. It starts from a calculated 'seed' point 
    between the window edges and expands outwards until the signal crosses 
    zero or changes sign, effectively "isolating" a single peak.
    
    Parameters:
    -----------
    coi : int
        Specific logic for overlapping waves. 
        0. find 0 point
        1: find next local minimum to the right; 
        2: find next local minimum to the left.
    '''
        
    # locate the seed
    t1, t2 = latsam
    segment = data[t1:t2+1]
    peak_r = np.argmax(np.abs(segment))
    seed = peak_r + t1
    peak_sign = np.sign(data[seed])
    if peak_sign == 0:
        return t1, t2
    
    # find zero-crossing
    a = seed
    while a > 0 and np.sign(data[a]) == peak_sign:
        a -= 1
    b = seed
    while b < len(data) - 1 and np.sign(data[b]) == peak_sign:
        b += 1
    
    if coi == 0:
        return a, b
        
    # overlapped waves
    data_rect = data.copy()
    data_rect[:a] = 0
    data_rect[b:] = 0
    data_rect = np.abs(data_rect)
    ndata = len(data_rect)
    
    # rectified data
    datamax = np.max(data_rect[a:b+1])
    imax = np.argmax(data_rect[a:b+1]) + a
    
    # find 10% onset/offset points
    ion_candi = np.where(data_rect[a:imax+1] > 0.2 * datamax)[0]
    ion = ion_candi[0] + a if len(ion_candi) > 0 else a 
    
    ioff_candi = np.where(data_rect[imax:b+1] > 0.2 * datamax)[0]
    ioff = ioff_candi[-1] + imax + 1 if len(ioff_candi) > 0 else b
    
    # ideal peak
    x_points = [a, ion, imax, ioff, b]
    y_points = [0, data_rect[ion], datamax, data_rect[ioff], 0]
    
    # interpolation
    xx = np.linspace(a, b, b-a+1)
    f_interp = interp1d(x_points, y_points, kind='cubic')
    simerp = f_interp(xx)
    
    # area threshold rate
    at1 = np.trapz(data_rect[a:b+1])
    at2 = np.trapz(simerp)
    atrate = at1 / at2 if at2 != 0 else 1
    
    if atrate < 0.9:
        step = 5
        for i in range(step, ndata - step):
            leftp = data_rect[max(0, i-step):i]
            rightp = data_rect[i+1 : i+1+step]
            targp = data_rect[i]
            
            con_l = np.sum(leftp > targp)
            con_r = np.sum(rightp >= targp)
            
            if con_l == step and con_r == step:
                if coi == 1:
                    b = i
                elif coi == 2:
                    a = i
                break
                
    return a, b
    

# fractional area latency
def _frac_area_latency(seg, tseg, area_tot, fraction, side):
    
    if area_tot == 0:
        return np.nan
    
    if side == 'left':
        cum_area = integrate.cumulative_trapezoid(seg, tseg, initial=0)
    elif side == 'right':
        cum_area = integrate.cumulative_trapezoid(seg[::-1], tseg[::-1], initial=0)[::-1]
    else:
        raise ValueError("`side` must be 'left' or 'right'.")
    
    target = fraction * area_tot
    idx = np.where(cum_area >= target)[0]
    frac_area = tseg[idx[0]] if len(idx) > 0 else np.nan
    
    return frac_area


def _area_calculation(seg, tseg, mode):
    
    seg_t = _apply_mode(seg, mode)
    area = integrate.trapezoid(seg_t, tseg)
    mean_amp = area / (tseg[-1] - tseg[0]) if tseg[-1] != tseg[0] else np.nan
    return area, mean_amp


def _apply_mode(seg, mode):
    
    if mode == "abs":
        return np.abs(seg)
    elif mode == "pos":
        return np.clip(seg, 0, None)   
    elif mode == "neg":
        return -np.clip(seg, None, 0)       
    elif mode == "intg":
        return seg  
    else:
        raise ValueError(f"Invalid mode: {mode}")
    

def _boundary_report(ch_name, old_indices, new_indices, timx):
    
    '''Report boundary change'''
    
    a_old, b_old = old_indices
    a_new, b_new = new_indices
    
    if a_old != a_new or b_old != b_new:
        return {
            "channel": ch_name,
            "orig_start": timx[a_old],
            "orig_end": timx[b_old],
            "new_start": timx[a_new],
            "new_end": timx[b_new],
            "shift_start_samples": a_new - a_old,
            "shift_end_samples": b_new - b_old            
        }
        
    return None