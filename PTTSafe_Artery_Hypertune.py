#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
 PPG/BP cleaning and synchronization tool

1. [ ] check_and_sync_signals:
   - Only repair when detecting large non-physiological delays (>0.5s)
   - Strictly preserve physiological PTT features in 0.08~0.5s (never align to 0!)
2. [Algorithm upgrade] Use Hilbert envelope for cross-correlation, more robust to noise than raw waveform
3. [Keep] 0.7-8Hz bandpass + forced inversion (-ppg)
4. [Keep] SBP median-filter based extraction
"""

import os
import numpy as np
import scipy.io as sio
from scipy import signal
from scipy.signal import find_peaks, peak_widths
from scipy.interpolate import interp1d
import json
from tqdm import tqdm
import argparse
import warnings
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import pickle

warnings.filterwarnings("ignore")

# ================= Configuration =================
DATA_DIR = "drink_process_data"
PPG_FS = 125
WINDOW_SEC = 15
STEP_SEC = 5
TARGET_LEN = 1024
ALL_BIOMARKERS = ['At', 'Dt', 'LASI', 'Cardiac_cycle', 'Pir', 'Asp', 'Dsp', 'Aa', 'Da', 'Aid', 'Did', 'Tpw50']

# ================= 1. Intelligent signal synchronization (Hilbert envelope version) =================
def check_and_sync_signals(ppg, ecg, fs=125, file_name="sample", max_shift_sec=2.0):
    """
    Detect delay using Hilbert envelope cross-correlation.
    - Physiological normal range (PPG lags ECG by 0.08~0.5s): do nothing — keep PTT.
    - Abnormal range (hardware desync): shift ECG for coarse alignment.
    """
    # Length/quality pre-check
    if len(ppg) < fs*5 or np.std(ecg) < 1e-3:
        return ecg

    # Extract envelopes (more stable, ignore waveform details, align energy peaks)
    # Use abs(hilbert(detrend(x)))
    p_env = np.abs(signal.hilbert(signal.detrend(ppg)))
    e_env = np.abs(signal.hilbert(signal.detrend(ecg)))

    # Compute cross-correlation: correlate(Ref, Target)
    # We take PPG as reference (Ref), ECG as target (Target)
    # The computed lag represents "how much ECG needs to move to match PPG"
    # Physiologically ECG comes first and PPG follows. So ECG should be shifted right (delayed) to align PPG
    # Expected normal lag is positive (around +0.2s * fs)
    corr = signal.correlate(p_env, e_env, mode='same')
    lags = signal.correlation_lags(len(p_env), len(e_env), mode='same')
    best_lag = lags[np.argmax(corr)]
    time_lag = best_lag / fs

    # === Key decision logic ===

    # Case A: Perfect physiological PTT range (0.08s ~ 0.5s)
    # This means ECG R-peaks indeed occur before PPG systolic peaks by ~200 ms — preserve it!
    if 0.08 <= time_lag <= 0.5:
        # print(f"  [Normal PTT] {file_name} lag={time_lag:.3f}s (preserve)")
        return ecg

    # Case B: Excessive delay (hardware desync)
    # e.g., time_lag = 3.0s (ECG is 3s early) or time_lag = -1.0s (ECG is later than PPG)
    # In such cases we correct, but to where?
    # Strategy: fully align (Lag -> 0). We lose PTT but salvage the sample usability.
    if abs(time_lag) > 0.5:  # threshold adjustable; outside 0.5s is considered abnormal
        # print(f"  [Fixing Sync] {file_name} lag={time_lag:.3f}s -> 0s")

        shift = int(best_lag)  # how much ECG should be shifted
        new_ecg = np.zeros_like(ecg)

        if shift > 0:
            # shift > 0: ECG needs to move right (ECG was originally too early)
            # pad at left (start)
            if shift < len(ecg):
                new_ecg[shift:] = ecg[:-shift]
        elif shift < 0:
            # shift < 0: ECG needs to move left (ECG was originally too late)
            # drop beginning, pad at end
            s = abs(shift)
            if s < len(ecg):
                new_ecg[:-s] = ecg[s:]

        return new_ecg

    # Other cases (0 < lag < 0.08): slightly close, not a big error, do nothing
    return ecg

# ================= 2. Core extraction and cleaning (keeping previous optimized logic) =================

def get_precise_sbp(bp_seg, fs=125):
    bp_clean = signal.medfilt(bp_seg, kernel_size=5)
    thresh = np.mean(bp_clean) + 10
    peaks, _ = find_peaks(bp_clean, distance=int(fs*0.4), prominence=8, height=thresh)
    if len(peaks) >= 2:
        return float(np.mean(bp_clean[peaks]))
    elif len(peaks) == 1:
        return float(bp_clean[peaks[0]])
    else:
        return float(np.percentile(bp_clean, 97))

def extract_biomarkers(ppg_segment, fs=125):
    if len(ppg_segment) < fs * 3:
        return [0.0] * 12

    # 1. Filtering + inversion (transmission PPG must be inverted!)
    b, a = signal.butter(4, [0.7, 8.0], btype='band', fs=fs)
    ppg = signal.filtfilt(b, a, ppg_segment.astype(float))
    ppg = signal.detrend(ppg)
    ppg = -ppg

    std = np.std(ppg)
    if std < 1e-8:
        return [0.0] * 12
    ppg = ppg / std

    peaks, _ = find_peaks(ppg, distance=int(0.6*fs), prominence=0.35, height=0.3)
    if len(peaks) < 2:
        return [0.0] * 12

    onsets, notches = [], []
    for i in range(1, len(peaks)):
        search_win = ppg[peaks[i-1]:peaks[i]]
        if len(search_win) == 0:
            continue
        onsets.append(peaks[i-1] + np.argmin(search_win))

        notch_win_start = peaks[i-1] + int(fs*0.10)
        notch_win_end = peaks[i-1] + int(fs*0.45)
        if notch_win_end > peaks[i]:
            notch_win_end = peaks[i]
        if notch_win_start < notch_win_end:
            win = ppg[notch_win_start:notch_win_end]
            if len(win) > 0:
                notches.append(notch_win_start + np.argmin(win))

    if not onsets:
        return [0.0] * 12

    # Feature calculations
    n = min(len(peaks)-1, len(onsets))
    if n <= 0:
        return [0.0]*12

    p_idxs = peaks[1:n+1]
    o_idxs = onsets[:n]

    At_list = (p_idxs - o_idxs) / float(fs)
    Dt_list = (p_idxs - peaks[:n]) / float(fs) - At_list
    At = float(np.mean(At_list)) if len(At_list) > 0 else 0.2
    Dt = float(np.mean(Dt_list)) if len(Dt_list) > 0 else 0.4

    cardiac_cycle = float(np.mean(np.diff(peaks))) / fs if len(peaks) > 1 else 0.8

    ph = ppg[p_idxs]
    oh = ppg[o_idxs]
    pulse_h = ph - oh

    if len(notches) >= n:
        n_idxs = np.array(notches)[:n]
        nh = ppg[n_idxs]
        Pir = float(np.mean(nh / (ph + 1e-8)))
        Asp = float(np.mean((ph - oh)/((p_idxs - o_idxs)/fs + 1e-8)))
        Dsp = float(np.mean((ph - nh)/((p_idxs - n_idxs)/fs + 1e-8)))
        Aa, Da = float(np.mean(pulse_h)), float(np.mean(ph - nh))
        Aid, Did = float(np.mean(pulse_h)*0.8), float(Da*1.2)
    else:
        Pir, Asp, Dsp, Aa, Da, Aid, Did = 0.5, 1.0, -0.5, np.mean(pulse_h), 0.5, 0.8, 0.6

    widths = peak_widths(ppg, peaks, rel_height=0.5)[0] / fs
    Tpw50 = float(np.mean(widths)) if len(widths) > 0 else 0.3
    LASI = float(np.std(ppg)/(np.mean(np.abs(ppg))+1e-8))

    feat = [At, Dt, LASI, cardiac_cycle, Pir, Asp, Dsp, Aa, Da, Aid, Did, Tpw50]
    return [0.0 if np.isnan(x) or np.isinf(x) else x for x in feat]

# ================= 3. Helpers and pipeline =================

def add_noise_and_augment(sig):
    noise = np.random.normal(0, 0.01, sig.shape)
    scale = np.random.uniform(0.95, 1.05)
    return sig * scale + noise

def detect_signals(mat_data):
    keys = [k for k in mat_data.keys() if not k.startswith('__')]
    return ('new_ppg' if 'new_ppg' in keys else None,
            'new_bp' if 'new_bp' in keys else None,
            'new_ecg' if 'new_ecg' in keys else None)

def generate_dataset(file_list, mode='train', fold_id=None):
    X_ppg_list, X_feat_list, X_ecg_list, Y_sbp_list = [], [], [], []
    save_dir = f"processed_{mode}_fold{fold_id}" if fold_id else "processed_dataset"
    os.makedirs(save_dir, exist_ok=True)

    for file_name in tqdm(file_list, desc=f"{mode} Fold {fold_id}" if fold_id else mode):
        path = os.path.join(DATA_DIR, file_name)
        if not os.path.exists(path):
            continue
        try:
            data = sio.loadmat(path)
        except:
            continue

        ppg_key, bp_key, ecg_key = detect_signals(data)
        if not all([ppg_key, bp_key]):
            continue

        ppg = np.array(data[ppg_key]).flatten()
        bp = np.array(data[bp_key]).flatten()
        ecg = np.array(data[ecg_key]).flatten() if ecg_key else np.zeros_like(ppg)
        min_len = min(len(ppg), len(bp), len(ecg))
        ppg, bp, ecg = ppg[:min_len], bp[:min_len], ecg[:min_len]

        # === [Repair] Intelligent ECG synchronization ===
        if ecg_key and np.std(ecg) > 1e-3:
            # pass file_name for debugging
            ecg = check_and_sync_signals(ppg, ecg, PPG_FS, file_name=file_name)

        window_samples = int(WINDOW_SEC * PPG_FS)
        step_samples = int(STEP_SEC * PPG_FS)

        for start in range(0, len(ppg) - window_samples + 1, step_samples):
            end = start + window_samples
            ppg_seg = ppg[start:end].astype(float)
            bp_seg = bp[start:end].astype(float)
            ecg_seg = ecg[start:end].astype(float)

            # Abnormality detection
            is_abnormal = (np.std(ppg_seg) < 1e-4 or np.mean(bp_seg) < 40 or np.mean(bp_seg) > 220)
            y_sbp = get_precise_sbp(bp_seg, PPG_FS)

            # Feature extraction
            feat_vec = extract_biomarkers(ppg_seg, PPG_FS)
            if feat_vec[0] == 0.0:
                is_abnormal = True

            # Resampling (with filter + invert)
            t_old = np.arange(len(ppg_seg)) / float(PPG_FS)
            t_new = np.linspace(0, WINDOW_SEC, TARGET_LEN)

            # PPG preprocessing
            b, a = signal.butter(4, [0.7, 8.0], btype='band', fs=PPG_FS)
            p_filt = signal.filtfilt(b, a, ppg_seg)
            p_filt = signal.detrend(p_filt) * -1

            f_ppg = interp1d(t_old, p_filt, kind='cubic', fill_value="extrapolate")
            ppg_res = f_ppg(t_new)

            # ECG preprocessing
            if np.std(ecg_seg) > 1e-3:
                b_e, a_e = signal.butter(4, [0.5, 40], btype='band', fs=PPG_FS)
                e_filt = signal.filtfilt(b_e, a_e, ecg_seg)
                f_ecg = interp1d(t_old, e_filt, kind='linear', fill_value="extrapolate")
                ecg_res = f_ecg(t_new)
            else:
                ecg_res = np.zeros(TARGET_LEN)

            # MinMax normalization
            def norm(x):
                mi, ma = np.min(x), np.max(x)
                return (x - mi)/(ma - mi + 1e-8) if ma-mi > 1e-8 else x

            ppg_res = norm(ppg_res)
            ecg_res = norm(ecg_res)

            if mode == 'train' and not is_abnormal:
                ppg_res = add_noise_and_augment(ppg_res)

            if not is_abnormal:
                X_ppg_list.append(ppg_res.astype(np.float32))
                X_feat_list.append(np.array(feat_vec, dtype=np.float32))
                X_ecg_list.append(ecg_res.astype(np.float32))
                Y_sbp_list.append(float(y_sbp))

    if not Y_sbp_list:
        print(f"Warning: {save_dir} empty")
        return

    X_ppg, X_feat, X_ecg, Y_sbp = np.array(X_ppg_list), np.array(X_feat_list), np.array(X_ecg_list), np.array(Y_sbp_list).reshape(-1,1)

    if mode == 'train':
        scaler = StandardScaler()
        X_feat = scaler.fit_transform(X_feat)
        with open(os.path.join(save_dir, "feat_scaler.pkl"), "wb") as f:
            pickle.dump(scaler, f)
    else:
        train_dir = f"processed_train_fold{fold_id}" if fold_id else "processed_dataset"
        scaler_path = os.path.join(train_dir, "feat_scaler.pkl")
        if os.path.exists(scaler_path):
            with open(scaler_path, "rb") as f:
                scaler = pickle.load(f)
            X_feat = scaler.transform(X_feat)
        else:
            scaler = StandardScaler()
            X_feat = scaler.fit_transform(X_feat)

    np.save(os.path.join(save_dir, "X_ppg.npy"), X_ppg)
    np.save(os.path.join(save_dir, "X_feat.npy"), X_feat)
    np.save(os.path.join(save_dir, "X_ecg.npy"), X_ecg)
    np.save(os.path.join(save_dir, "Y_sbp.npy"), Y_sbp)
    print(f"✓ {save_dir} | Count: {len(Y_sbp)}")

def run_cv_preparation():
    if not os.path.exists('cv_15_5_5.json'):
        return
    with open('cv_15_5_5.json') as f:
        folds = json.load(f)
    for fold in folds:
        generate_dataset(fold.get('train', []), 'train', fold['fold'])
        generate_dataset(fold.get('val', []), 'val', fold['fold'])
        generate_dataset(fold.get('test', []), 'test', fold['fold'])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--cv', action='store_true')
    args = parser.parse_args()
    if args.cv:
        run_cv_preparation()
    else:
        files = sorted([f for f in os.listdir(DATA_DIR) if f.endswith('.mat')])
        if files:
            split = int(0.8 * len(files))
            generate_dataset(files[:split], 'train')
            generate_dataset(files[split:], 'test')
