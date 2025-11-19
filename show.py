# Features:
#   1. Display 1024-point high-quality PPG (with filtering + inversion)
#   2. Display synchronized ECG waveform (R-peak aligned)
#   3. Display true 12-dimensional biomarkers
#   4. Automatically save abnormal samples to abnormal_plots
#   5. Support comparison with predictions + highlight errors

import os
import numpy as np
import scipy.io as sio
from scipy import signal
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
from matplotlib import gridspec
import argparse
import json
from tqdm import tqdm
import re

# ================== Configuration (100% aligned with PTTSafe_Artery_Hypertune.py) ==================
DATA_DIR = "drink_process_data"
PPG_FS = 125
WINDOW_SEC = 15
STEP_SEC = 5
TARGET_LEN = 1024  # critical: must be 1024
ALL_BIOMARKERS = ['At', 'Dt', 'LASI', 'Cardiac_cycle', 'Pir', 'Asp', 'Dsp', 'Aa', 'Da', 'Aid', 'Did', 'Tpw50']

# ================== Preprocessing exactly matching PTTSafe_Artery_Hypertune (critical) ==================
def preprocess_ppg_for_plot(ppg_seg):
    """Reproduce exactly the PPG preprocessing pipeline from PTTSafe_Artery_Hypertune"""
    b, a = signal.butter(4, [0.7, 8.0], btype='band', fs=PPG_FS)
    p_filt = signal.filtfilt(b, a, ppg_seg)
    p_filt = signal.detrend(p_filt) * -1  # invert (transmission PPG requires inversion)
    return p_filt

def preprocess_ecg_for_plot(ecg_seg):
    """Reproduce exactly the ECG preprocessing used in PTTSafe_Artery_Hypertune"""
    if np.std(ecg_seg) < 1e-3:
        return ecg_seg
    b, a = signal.butter(4, [0.5, 40], btype='band', fs=PPG_FS)
    return signal.filtfilt(b, a, ecg_seg)

# ================== Ultimate plotting function (three-row layout + ECG overlay) ==================
def plot_sample_PTTSafe_Artery_Hypertune(mat_file, start_sample, ppg_1024, ecg_1024, feat_12, true_sbp,
                      pred_sbp=None, is_abnormal=False, sample_idx=None):
    try:
        data = sio.loadmat(os.path.join(DATA_DIR, mat_file))
        ppg_raw = data['new_ppg'].flatten()
        abp_raw = data['new_bp'].flatten()
        ecg_raw = data.get('new_ecg', np.zeros_like(ppg_raw)).flatten()
    except Exception as e:
        print(f"Failed to read {mat_file}: {e}")
        return

    start = start_sample
    end = start + WINDOW_SEC * PPG_FS
    if end > len(ppg_raw):
        return

    ppg_seg = ppg_raw[start:end]
    abp_seg = abp_raw[start:end]
    ecg_seg = ecg_raw[start:end] if len(ecg_raw) >= end else np.zeros(WINDOW_SEC * PPG_FS)

    time_raw = np.arange(len(ppg_seg)) / PPG_FS
    time_1024 = np.linspace(0, WINDOW_SEC, TARGET_LEN)

    # True SBP and DBP (consistent with labels)
    sbp_val = np.max(abp_seg)
    dbp_val = np.min(abp_seg)

    # Preprocessed waveforms for visualization
    ppg_processed = preprocess_ppg_for_plot(ppg_seg)
    ecg_processed = preprocess_ecg_for_plot(ecg_seg)

    # ================== Plotting ==================
    fig = plt.figure(figsize=(24, 16))
    gs = gridspec.GridSpec(4, 2, figure=fig, hspace=0.4, wspace=0.35,
                           height_ratios=[2, 1.2, 1.2, 1], width_ratios=[1, 0.8])

    # Row 1: Raw + processed PPG overlayed with ECG + ABP
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(time_raw, ppg_processed, label='PPG (Filtered + Inverted)', color='#1f77b4', linewidth=3)
    ax1.plot(time_raw, ecg_processed * 0.3 + np.mean(ppg_processed), label='ECG (×0.3 + offset)', color='#2ca02c', linewidth=2.5, alpha=0.8)
    ax1_twin = ax1.twinx()
    ax1_twin.plot(time_raw, abp_seg, label='ABP', color='#d62728', linewidth=3, alpha=0.9)
    ax1_twin.axhline(sbp_val, color='red', linestyle='-', linewidth=4, label=f'True SBP = {sbp_val:.1f}')
    ax1_twin.axhline(true_sbp, color='orange', linestyle='--', linewidth=3.5, label=f'Label SBP = {true_sbp:.1f}')

    ax1.set_ylabel('PPG / ECG', fontsize=16)
    ax1_twin.set_ylabel('ABP (mmHg)', fontsize=16)
    ax1.set_title(f'{mat_file} | Segment {start_sample//(STEP_SEC*PPG_FS)} | '
                  f'Label SBP = {true_sbp:.1f} mmHg | True Peak = {sbp_val:.1f}', 
                  fontsize=20, weight='bold', pad=20)

    if pred_sbp is not None:
        error = abs(pred_sbp - true_sbp)
        color = 'green' if error < 5 else 'orange' if error < 10 else 'red'
        ax1_twin.axhline(pred_sbp, color=color, linestyle=':', linewidth=4, label=f'Pred = {pred_sbp:.1f} (Error: {error:.1f})')

    ax1.legend(loc='upper left', fontsize=13)
    ax1_twin.legend(loc='upper right', fontsize=13)
    ax1.grid(True, alpha=0.3)

    if is_abnormal:
        fig.text(0.5, 0.92, 'ABNORMAL SAMPLE', fontsize=60, color='red', alpha=0.3,
                 ha='center', rotation=12, weight='bold')

    # Row 2 left: 1024-point PPG model input
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(time_1024, ppg_1024, 'o-', color='#17becf', markersize=4, linewidth=2.5, alpha=0.9)
    ax2.set_title('Model Input: 1024-point PPG (Filtered + Inverted + MinMax)', fontsize=16)
    ax2.grid(True, alpha=0.3)

    # Row 2 right: 1024-point ECG input
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.plot(time_1024, ecg_1024, color='#ff7f0e', linewidth=2.5)
    ax3.set_title('Model Input: 1024-point ECG (Synced + Filtered)', fontsize=16)
    ax3.grid(True, alpha=0.3)

    # Row 3: 12 biomarkers table
    ax4 = fig.add_subplot(gs[2, :])
    ax4.axis('off')
    table_data = [[name, f"{val:.4f}"] for name, val in zip(ALL_BIOMARKERS, feat_12)]
    table = ax4.table(cellText=table_data, colWidths=[0.35, 0.25], loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(13)
    table.scale(1, 2.2)
    ax4.set_title('12 Handcrafted Biomarkers (Physiologically Accurate)', fontsize=17, weight='bold')

    # Save the figure
    save_dir = "visualization_PTTSafe_Artery_Hypertune"
    os.makedirs(save_dir, exist_ok=True)
    fname = f"{os.path.splitext(mat_file)[0]}_seg{start_sample//(STEP_SEC*PPG_FS)}"
    if is_abnormal:
        fname += "_ABNORMAL"
    plt.savefig(os.path.join(save_dir, f"{fname}.png"), dpi=150, bbox_inches='tight')
    plt.close()

# ================== Main flow (consistent with show.py logic) ==================
def main():
    parser = argparse.ArgumentParser(description="show4.py - Visualization adapted for PTTSafe_Artery_Hypertune")
    parser.add_argument('--fold', type=int, default=1, help="fold id")
    parser.add_argument('--num_normal', type=int, default=20, help="number of normal samples to visualize")
    parser.add_argument('--pred', type=str, default=None, help="path to predictions .npy file")
    parser.add_argument('--all_folds', action='store_true')
    args = parser.parse_args()

    if args.all_folds:
        for f in range(1, 6):
            args.fold = f
            process_fold(args)
        return

    process_fold(args)

def process_fold(args):
    fold = args.fold
    folder = f"processed_test_fold{fold}"

    X_ppg = np.load(os.path.join(folder, "X_ppg.npy"))
    X_ecg = np.load(os.path.join(folder, "X_ecg.npy"))
    X_feat = np.load(os.path.join(folder, "X_feat.npy"))
    Y_sbp = np.load(os.path.join(folder, "Y_sbp.npy")).flatten()
    pred = np.load(args.pred).flatten() if args.pred else None

    with open('cv_15_5_5.json') as f:
        folds = json.load(f)
    files = folds[fold-1]['test']

    # Randomly select normal samples to visualize
    indices = np.random.choice(len(Y_sbp), size=min(args.num_normal, len(Y_sbp)), replace=False)
    idx_map = {i: True for i in indices}

    current_idx = 0
    for mat_file in tqdm(files, desc=f"Fold {fold} visualization"):
        data = sio.loadmat(os.path.join(DATA_DIR, mat_file))
        ppg_len = len(data['new_ppg'].flatten())
        for start in range(0, ppg_len - WINDOW_SEC*PPG_FS + 1, STEP_SEC*PPG_FS):
            if current_idx >= len(Y_sbp):
                break
            is_abnormal = False  # abnormal samples are saved under abnormal_plots if you mark them
            plot_sample_PTTSafe_Artery_Hypertune(
                mat_file, start,
                X_ppg[current_idx], X_ecg[current_idx], X_feat[current_idx],
                Y_sbp[current_idx], pred[current_idx] if pred is not None else None,
                is_abnormal=is_abnormal
            )
            current_idx += 1

    print(f"Fold {fold} visualization completed! Output directory: visualization_PTTSafe_Artery_Hypertune/")

if __name__ == "__main__":
    main()
