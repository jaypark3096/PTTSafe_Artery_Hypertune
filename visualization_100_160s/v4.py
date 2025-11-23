import os
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import argparse
from tqdm import tqdm
import pandas as pd
from scipy.signal import find_peaks, butter, filtfilt
import warnings
warnings.filterwarnings('ignore')

# ================= 配置参数 =================
DATA_DIR = "drink_process_data"
OUTPUT_DIR = "visualization_100_160s"
CSV_FILENAME = "signal_metrics_100_160s.csv"
PPG_FS = 125
START_TIME = 100  # 开始时间：100秒
END_TIME = 160    # 结束时间：160秒
PLOT_DURATION = END_TIME - START_TIME  # 总共60秒

# 高血压和低血压患者的文件名关键词（根据实际情况调整）
HYPERTENSION_KEYWORDS = ['hypertension', 'high_bp', 'htn', 'high']  # 高血压
HYPOTENSION_KEYWORDS = ['hypotension', 'low_bp', 'low']  # 低血压

# ================= 创建输出目录 =================
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ================= 检测信号函数 =================
def detect_signals(mat_data):
    """检测MAT文件中存在的信号类型"""
    keys = [k for k in mat_data.keys() if not k.startswith('__')]
    ppg_key = 'new_ppg' if 'new_ppg' in keys else None
    bp_key = 'new_bp' if 'new_bp' in keys else None
    ecg_key = 'new_ecg' if 'new_ecg' in keys else None
    return ppg_key, bp_key, ecg_key

# ================= 检查是否为正常血压患者 =================
def is_normal_bp_patient(filename):
    """检查文件名是否包含高血压或低血压关键词"""
    filename_lower = filename.lower()
    
    # 检查高血压关键词
    for keyword in HYPERTENSION_KEYWORDS:
        if keyword in filename_lower:
            return False
    
    # 检查低血压关键词
    for keyword in HYPOTENSION_KEYWORDS:
        if keyword in filename_lower:
            return False
    
    return True

# ================= 信号预处理 =================
def preprocess_signal(signal_data, signal_type='ppg'):
    """对信号进行预处理"""
    if len(signal_data) == 0:
        return signal_data
        
    if signal_type == 'ppg':
        # PPG信号：带通滤波 + 反转（透射式）
        b, a = butter(4, [0.7, 8.0], btype='band', fs=PPG_FS)
        filtered = filtfilt(b, a, signal_data.astype(float))
        filtered = -filtered  # 反转信号
        return filtered
    elif signal_type == 'ecg':
        # ECG信号：带通滤波
        b, a = butter(4, [0.5, 40], btype='band', fs=PPG_FS)
        filtered = filtfilt(b, a, signal_data.astype(float))
        return filtered
    else:
        # BP信号：轻微平滑
        from scipy import signal
        return signal.medfilt(signal_data, kernel_size=3)

# ================= 提取100-160秒数据 =================
def extract_100_160_segment(signal_data, fs=125, start_time=100, end_time=160):
    """提取100秒到160秒的数据段"""
    start_sample = int(start_time * fs)
    end_sample = int(end_time * fs)
    
    if len(signal_data) < end_sample:
        # 如果数据长度不足，尝试使用最后60秒
        if len(signal_data) >= 60 * fs:
            start_sample = len(signal_data) - 60 * fs
            end_sample = len(signal_data)
        else:
            return np.array([])  # 数据太短
    
    return signal_data[start_sample:end_sample]

# ================= 计算信号统计指标 =================
def calculate_signal_metrics(signal_data, signal_name, fs=125):
    """计算信号的详细统计指标"""
    if len(signal_data) == 0:
        return {}
    
    metrics = {
        f'{signal_name}_mean': np.mean(signal_data),
        f'{signal_name}_std': np.std(signal_data),
        f'{signal_name}_min': np.min(signal_data),
        f'{signal_name}_max': np.max(signal_data),
        f'{signal_name}_range': np.max(signal_data) - np.min(signal_data),
        f'{signal_name}_length': len(signal_data)
    }
    
    # 信号特定的额外指标
    if signal_name == 'PPG':
        # PPG特定指标：峰值计数、心率变异性等
        peaks, _ = find_peaks(signal_data, distance=int(fs*0.4), prominence=0.1)
        metrics['PPG_peak_count'] = len(peaks)
        if len(peaks) > 1:
            rr_intervals = np.diff(peaks) / fs
            metrics['PPG_heart_rate'] = 60 / np.mean(rr_intervals) if np.mean(rr_intervals) > 0 else 0
            metrics['PPG_hrv_std'] = np.std(rr_intervals)
        else:
            metrics['PPG_heart_rate'] = 0
            metrics['PPG_hrv_std'] = 0
    
    elif signal_name == 'ECG':
        # ECG特定指标：R峰计数、心率等
        peaks, _ = find_peaks(signal_data, distance=int(fs*0.4), prominence=0.5)
        metrics['ECG_R_peak_count'] = len(peaks)
        if len(peaks) > 1:
            rr_intervals = np.diff(peaks) / fs
            metrics['ECG_heart_rate'] = 60 / np.mean(rr_intervals) if np.mean(rr_intervals) > 0 else 0
            metrics['ECG_hrv_std'] = np.std(rr_intervals)
        else:
            metrics['ECG_heart_rate'] = 0
            metrics['ECG_hrv_std'] = 0
    
    elif signal_name == 'BP':
        # BP特定指标：收缩压、舒张压、脉压等
        peaks, properties = find_peaks(signal_data, distance=int(fs*0.4), prominence=10)
        
        if len(peaks) > 0:
            systolic_bp = np.mean(signal_data[peaks])
            metrics['BP_systolic'] = systolic_bp
            
            # 寻找舒张压（峰值之间的最低点）
            diastolic_values = []
            for i in range(len(peaks)-1):
                valley = np.min(signal_data[peaks[i]:peaks[i+1]])
                diastolic_values.append(valley)
            
            diastolic_bp = np.mean(diastolic_values) if diastolic_values else np.percentile(signal_data, 30)
            metrics['BP_diastolic'] = diastolic_bp
            metrics['BP_pulse_pressure'] = systolic_bp - diastolic_bp
            metrics['BP_mean'] = np.mean(signal_data)
            metrics['BP_peak_count'] = len(peaks)
            metrics['BP_heart_rate'] = len(peaks) / (len(signal_data) / fs) * 60
            
            # 血压分类
            if systolic_bp < 90 or diastolic_bp < 60:
                metrics['BP_category'] = 'Hypotension'
            elif systolic_bp >= 140 or diastolic_bp >= 90:
                metrics['BP_category'] = 'Hypertension'
            else:
                metrics['BP_category'] = 'Normal'
        else:
            metrics['BP_systolic'] = np.mean(signal_data)
            metrics['BP_diastolic'] = np.percentile(signal_data, 30)
            metrics['BP_pulse_pressure'] = metrics['BP_systolic'] - metrics['BP_diastolic']
            metrics['BP_mean'] = np.mean(signal_data)
            metrics['BP_peak_count'] = 0
            metrics['BP_heart_rate'] = 0
            metrics['BP_category'] = 'Unknown'
    
    return metrics

# ================= 评估信号质量 =================
def evaluate_signal_quality(signal_data, signal_name, fs=125):
    """评估信号质量并返回质量等级和原因"""
    if len(signal_data) == 0:
        return "NO_DATA", "数据长度不足"
    
    # 检查信号幅度
    signal_range = np.max(signal_data) - np.min(signal_data)
    
    if signal_name == 'PPG':
        if signal_range < 0.1:
            return "POOR", "信号幅度太小"
        elif signal_range > 10:
            return "POOR", "信号幅度过大"
        else:
            # 检查是否有明显的周期性
            peaks, _ = find_peaks(signal_data, distance=int(fs*0.4), prominence=0.1)
            if len(peaks) < 2:
                return "POOR", "周期性不明显"
            else:
                return "GOOD", "信号质量良好"
    
    elif signal_name == 'ECG':
        if signal_range < 0.1:
            return "POOR", "信号幅度太小"
        else:
            # 检查是否有明显的R峰
            peaks, _ = find_peaks(signal_data, distance=int(fs*0.4), prominence=0.5)
            if len(peaks) < 2:
                return "POOR", "R峰检测失败"
            else:
                return "GOOD", "信号质量良好"
    
    elif signal_name == 'BP':
        mean_bp = np.mean(signal_data)
        if mean_bp < 50:
            return "POOR", f"血压值过低: {mean_bp:.1f}"
        elif mean_bp > 200:
            return "POOR", f"血压值过高: {mean_bp:.1f}"
        else:
            peaks, _ = find_peaks(signal_data, distance=int(fs*0.4), prominence=10)
            if len(peaks) < 2:
                return "SUSPICIOUS", "血压峰值检测较少"
            else:
                return "GOOD", "血压信号良好"
    
    return "UNKNOWN", "未知质量"

# ================= 收集所有指标 =================
def collect_all_metrics(filename, signals_dict, fs=125):
    """收集文件的所有信号指标"""
    metrics = {'filename': filename}
    
    # 为每个信号计算指标
    for signal_name, signal_data in signals_dict.items():
        signal_metrics = calculate_signal_metrics(signal_data, signal_name, fs)
        metrics.update(signal_metrics)
        
        # 添加质量评估
        quality, reason = evaluate_signal_quality(signal_data, signal_name, fs)
        metrics[f'{signal_name}_quality'] = quality
        metrics[f'{signal_name}_quality_reason'] = reason
    
    # 计算总体质量评分
    quality_scores = {'GOOD': 2, 'SUSPICIOUS': 1, 'POOR': 0, 'NO_DATA': -1}
    signal_count = len(signals_dict)
    if signal_count > 0:
        total_score = sum(quality_scores.get(metrics.get(f'{sig}_quality', 'UNKNOWN'), 0) 
                         for sig in signals_dict.keys())
        metrics['overall_quality_score'] = total_score / signal_count
        metrics['signal_count'] = signal_count
    else:
        metrics['overall_quality_score'] = -1
        metrics['signal_count'] = 0
    
    return metrics

# ================= 可视化单个文件 =================
def visualize_file(file_path, output_path):
    """可视化单个MAT文件的100-160秒信号并返回指标"""
    try:
        # 加载数据
        data = sio.loadmat(file_path)
        
        # 检测信号
        ppg_key, bp_key, ecg_key = detect_signals(data)
        
        # 提取信号数据
        signals = {}
        raw_signals = {}  # 用于指标计算的原始信号
        
        if ppg_key:
            ppg_raw = np.array(data[ppg_key]).flatten()
            ppg_segment = extract_100_160_segment(ppg_raw)
            if len(ppg_segment) > 0:
                signals['PPG'] = preprocess_signal(ppg_segment, 'ppg')
                raw_signals['PPG'] = ppg_segment
        
        if bp_key:
            bp_raw = np.array(data[bp_key]).flatten()
            bp_segment = extract_100_160_segment(bp_raw)
            if len(bp_segment) > 0:
                signals['BP'] = preprocess_signal(bp_segment, 'bp')
                raw_signals['BP'] = bp_segment
        
        if ecg_key:
            ecg_raw = np.array(data[ecg_key]).flatten()
            ecg_segment = extract_100_160_segment(ecg_raw)
            if len(ecg_segment) > 0:
                signals['ECG'] = preprocess_signal(ecg_segment, 'ecg')
                raw_signals['ECG'] = ecg_segment
        
        if not signals:
            print(f"  No valid signals found in {os.path.basename(file_path)} (100-160s)")
            return False, "NO_VALID_DATA", {}
        
        # 收集指标
        metrics = collect_all_metrics(os.path.basename(file_path), raw_signals, PPG_FS)
        
        # 确定要绘制的样本数
        plot_samples = min(len(signal) for signal in signals.values())
        
        if plot_samples < PPG_FS:  # 至少要有1秒数据
            print(f"  Insufficient data in {os.path.basename(file_path)}")
            return False, "INSUFFICIENT_DATA", metrics
        
        # 创建图形
        fig = plt.figure(figsize=(16, 12))
        gs = GridSpec(len(signals), 1, figure=fig)
        
        # 绘制每个信号
        stats_text = []
        quality_text = []
        
        for i, (signal_name, signal_data) in enumerate(signals.items()):
            # 提取对应长度的数据
            segment = signal_data[:plot_samples]
            
            # 创建子图
            ax = fig.add_subplot(gs[i])
            
            # 生成时间轴 (100-160秒)
            time_axis = np.arange(plot_samples) / PPG_FS + START_TIME
            
            # 设置颜色
            colors = {
                'PPG': 'blue',
                'BP': 'red', 
                'ECG': 'green'
            }
            color = colors.get(signal_name, 'black')
            
            # 绘制信号
            ax.plot(time_axis, segment, linewidth=1.0, color=color)
            
            # 设置子图属性
            ax.set_title(f'{signal_name} Signal - {os.path.basename(file_path)} ({START_TIME}s-{END_TIME}s)', 
                        fontsize=14, fontweight='bold')
            ax.set_ylabel('Amplitude', fontsize=12)
            ax.grid(True, alpha=0.3)
            ax.set_xlim(START_TIME, START_TIME + plot_samples / PPG_FS)
            
            # 添加时间网格
            ax.set_xticks(np.arange(START_TIME, START_TIME + PLOT_DURATION + 1, 10))
            ax.grid(True, alpha=0.2)
            
            # 添加质量评估
            quality = metrics.get(f'{signal_name}_quality', 'UNKNOWN')
            reason = metrics.get(f'{signal_name}_quality_reason', 'Unknown reason')
            color_map = {'GOOD': 'green', 'POOR': 'red', 'SUSPICIOUS': 'orange', 'UNKNOWN': 'gray'}
            quality_color = color_map.get(quality, 'black')
            ax.text(0.02, 0.98, f'Quality: {quality}', transform=ax.transAxes, 
                   color=quality_color, fontweight='bold', verticalalignment='top',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor=quality_color, alpha=0.2))
            quality_text.append(f"{signal_name}: {quality} - {reason}")
            
            # 添加关键指标到子图
            if signal_name == 'BP':
                sbp = metrics.get('BP_systolic', 0)
                dbp = metrics.get('BP_diastolic', 0)
                hr = metrics.get('BP_heart_rate', 0)
                stats_text.append(f"SBP: {sbp:.1f}, DBP: {dbp:.1f}, HR: {hr:.1f} bpm")
            elif signal_name == 'PPG':
                hr = metrics.get('PPG_heart_rate', 0)
                peaks = metrics.get('PPG_peak_count', 0)
                stats_text.append(f"HR: {hr:.1f} bpm, Peaks: {peaks}")
            elif signal_name == 'ECG':
                hr = metrics.get('ECG_heart_rate', 0)
                peaks = metrics.get('ECG_R_peak_count', 0)
                stats_text.append(f"HR: {hr:.1f} bpm, R-peaks: {peaks}")
        
        # 设置x轴标签（只在最后一个子图）
        ax.set_xlabel('Time (seconds)', fontsize=12)
        
        # 添加统计信息文本框
        stats_str = '\n'.join(stats_text)
        fig.text(0.02, 0.02, stats_str, fontsize=10, verticalalignment='bottom',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))
        
        # 添加质量评估文本框
        quality_str = '\n'.join(quality_text)
        fig.text(0.6, 0.02, quality_str, fontsize=9, verticalalignment='bottom',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8))
        
        # 添加总标题
        plt.suptitle(f'Signal Visualization - {START_TIME}s to {END_TIME}s (Device Delay Check)', 
                    fontsize=16, fontweight='bold', y=0.98)
        
        # 调整布局并保存
        plt.tight_layout(rect=[0, 0.08, 1, 0.96])
        plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        
        # 汇总质量评估
        overall_quality = "GOOD"
        if any(metrics.get(f'{sig}_quality') == "POOR" for sig in signals.keys()):
            overall_quality = "POOR"
        elif any(metrics.get(f'{sig}_quality') == "SUSPICIOUS" for sig in signals.keys()):
            overall_quality = "SUSPICIOUS"
        
        print(f"  ✓ Saved: {os.path.basename(output_path)} - Quality: {overall_quality}")
        return True, overall_quality, metrics
        
    except Exception as e:
        print(f"  Error processing {file_path}: {str(e)}")
        return False, "ERROR", {}

# ================= 保存指标到CSV =================
def save_metrics_to_csv(all_metrics, csv_path):
    """将所有指标保存到CSV文件"""
    if not all_metrics:
        print("No metrics to save!")
        return
    
    # 创建DataFrame
    df = pd.DataFrame(all_metrics)
    
    # 重新排列列，将重要指标放在前面
    preferred_order = ['filename', 'overall_quality_score', 'signal_count']
    signal_types = ['PPG', 'BP', 'ECG']
    
    for signal in signal_types:
        preferred_order.extend([
            f'{signal}_quality', f'{signal}_quality_reason',
            f'{signal}_mean', f'{signal}_std', f'{signal}_min', f'{signal}_max', f'{signal}_range'
        ])
        
        # 添加信号特定指标
        if signal == 'BP':
            preferred_order.extend(['BP_systolic', 'BP_diastolic', 'BP_pulse_pressure', 
                                  'BP_mean', 'BP_peak_count', 'BP_heart_rate', 'BP_category'])
        elif signal == 'PPG':
            preferred_order.extend(['PPG_peak_count', 'PPG_heart_rate', 'PPG_hrv_std'])
        elif signal == 'ECG':
            preferred_order.extend(['ECG_R_peak_count', 'ECG_heart_rate', 'ECG_hrv_std'])
    
    # 重新排列列顺序
    existing_columns = [col for col in preferred_order if col in df.columns]
    other_columns = [col for col in df.columns if col not in preferred_order]
    df = df[existing_columns + other_columns]
    
    # 保存到CSV
    df.to_csv(csv_path, index=False, encoding='utf-8')
    print(f"Metrics saved to: {csv_path}")
    
    # 显示基本统计
    print(f"\nCSV Summary:")
    print(f"  Total records: {len(df)}")
    print(f"  Columns: {len(df.columns)}")
    
    # 显示质量分布
    if 'overall_quality_score' in df.columns:
        print(f"  Average quality score: {df['overall_quality_score'].mean():.2f}")
    
    return df

# ================= 生成汇总报告 =================
def generate_summary_report(results, csv_df):
    """生成处理汇总报告"""
    report_path = os.path.join(OUTPUT_DIR, "device_delay_check_summary.txt")
    
    # 统计各类质量的文件数量
    quality_counts = {
        'GOOD': 0,
        'SUSPICIOUS': 0,
        'POOR': 0,
        'ERROR': 0,
        'NO_VALID_DATA': 0,
        'INSUFFICIENT_DATA': 0
    }
    
    for filename, success, quality, _ in results:
        if success:
            quality_counts[quality] += 1
        else:
            quality_counts[quality] += 1
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=== Device Delay Check Summary Report ===\n\n")
        f.write(f"Data directory: {DATA_DIR}\n")
        f.write(f"Output directory: {OUTPUT_DIR}\n")
        f.write(f"Analysis time window: {START_TIME}s to {END_TIME}s\n")
        f.write(f"CSV file: {CSV_FILENAME}\n")
        f.write(f"Total files processed: {len(results)}\n\n")
        
        f.write("Quality Summary:\n")
        f.write(f"  GOOD: {quality_counts['GOOD']} files\n")
        f.write(f"  SUSPICIOUS: {quality_counts['SUSPICIOUS']} files\n")
        f.write(f"  POOR: {quality_counts['POOR']} files\n")
        f.write(f"  ERROR: {quality_counts['ERROR']} files\n")
        f.write(f"  NO_VALID_DATA: {quality_counts['NO_VALID_DATA']} files\n")
        f.write(f"  INSUFFICIENT_DATA: {quality_counts['INSUFFICIENT_DATA']} files\n\n")
        
        if csv_df is not None and not csv_df.empty:
            f.write("Key Metrics Summary:\n")
            if 'BP_systolic' in csv_df.columns:
                f.write(f"  Average SBP: {csv_df['BP_systolic'].mean():.1f} ± {csv_df['BP_systolic'].std():.1f}\n")
            if 'BP_diastolic' in csv_df.columns:
                f.write(f"  Average DBP: {csv_df['BP_diastolic'].mean():.1f} ± {csv_df['BP_diastolic'].std():.1f}\n")
            if 'PPG_heart_rate' in csv_df.columns:
                f.write(f"  Average PPG HR: {csv_df['PPG_heart_rate'].mean():.1f} ± {csv_df['PPG_heart_rate'].std():.1f}\n")
            if 'ECG_heart_rate' in csv_df.columns:
                # 修正：使用csv_df而不是df
                valid_ecg_hr = csv_df[csv_df['ECG_heart_rate'] > 0]['ECG_heart_rate']
                if len(valid_ecg_hr) > 0:
                    f.write(f"  Average ECG HR: {valid_ecg_hr.mean():.1f} ± {valid_ecg_hr.std():.1f}\n")
                else:
                    f.write(f"  Average ECG HR: No valid ECG heart rate data\n")
            f.write("\n")
        
        f.write("Detailed Results:\n")
        for filename, success, quality, _ in results:
            status = "SUCCESS" if success else "FAILED"
            f.write(f"  {filename}: {status} - {quality}\n")
    
    print(f"Summary report saved: {report_path}")
    return quality_counts

# ================= 主函数 =================
def main():
    """主函数：处理所有正常血压患者的MAT文件，分析100-160秒数据并保存指标到CSV"""
    # 获取所有MAT文件
    all_files = [f for f in os.listdir(DATA_DIR) if f.endswith('.mat')]
    
    if not all_files:
        print(f"No .mat files found in {DATA_DIR}")
        return
    
    print(f"Found {len(all_files)} .mat files in {DATA_DIR}")
    
    # 过滤出正常血压患者的文件
    normal_bp_files = [f for f in all_files if is_normal_bp_patient(f)]
    
    print(f"Normal BP patients: {len(normal_bp_files)} files")
    print(f"Excluded files (hypertension/hypotension): {len(all_files) - len(normal_bp_files)} files")
    
    if not normal_bp_files:
        print("No normal BP patient files found!")
        return
    
    # 显示被排除的文件（用于调试）
    excluded_files = [f for f in all_files if not is_normal_bp_patient(f)]
    if excluded_files:
        print(f"\nExcluded files: {excluded_files}")
    
    # 可视化每个文件的100-160秒数据并收集指标
    print(f"\nAnalyzing {START_TIME}s to {END_TIME}s segment for device delay check...")
    results = []
    all_metrics = []
    
    for filename in tqdm(normal_bp_files, desc="Processing files"):
        file_path = os.path.join(DATA_DIR, filename)
        output_filename = os.path.splitext(filename)[0] + f'_{START_TIME}_{END_TIME}s.png'
        output_path = os.path.join(OUTPUT_DIR, output_filename)
        
        success, quality, metrics = visualize_file(file_path, output_path)
        results.append((filename, success, quality, metrics))
        
        # 只收集成功处理的文件的指标
        if success and metrics:
            all_metrics.append(metrics)
    
    # 保存指标到CSV
    csv_path = os.path.join(OUTPUT_DIR, CSV_FILENAME)
    csv_df = save_metrics_to_csv(all_metrics, csv_path)
    
    # 生成汇总报告
    quality_counts = generate_summary_report(results, csv_df)
    
    print(f"\n{'='*60}")
    print(f"DEVICE DELAY CHECK COMPLETED!")
    print(f"{'='*60}")
    print(f"Time window analyzed: {START_TIME}s to {END_TIME}s")
    print(f"Total files processed: {len(normal_bp_files)}")
    print(f"Successfully visualized: {quality_counts['GOOD'] + quality_counts['SUSPICIOUS']}")
    print(f"Metrics saved to: {csv_path}")
    print(f"\nQuality Breakdown:")
    print(f"  GOOD: {quality_counts['GOOD']} files")
    print(f"  SUSPICIOUS: {quality_counts['SUSPICIOUS']} files") 
    print(f"  POOR: {quality_counts['POOR']} files")
    print(f"  Other issues: {quality_counts['ERROR'] + quality_counts['NO_VALID_DATA'] + quality_counts['INSUFFICIENT_DATA']} files")
    print(f"\nOutput directory: {OUTPUT_DIR}")
    print(f"{'='*60}")

if __name__ == "__main__":
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='Check device delay by analyzing 100-160s signal segments and save metrics to CSV')
    parser.add_argument('--data_dir', type=str, default=DATA_DIR, help='Input data directory')
    parser.add_argument('--output_dir', type=str, default=OUTPUT_DIR, help='Output directory for PNG files')
    parser.add_argument('--start_time', type=int, default=START_TIME, help='Start time in seconds')
    parser.add_argument('--end_time', type=int, default=END_TIME, help='End time in seconds')
    parser.add_argument('--csv_filename', type=str, default=CSV_FILENAME, help='Output CSV filename')
    
    args = parser.parse_args()
    
    # 更新配置
    DATA_DIR = args.data_dir
    OUTPUT_DIR = args.output_dir
    START_TIME = args.start_time
    END_TIME = args.end_time
    CSV_FILENAME = args.csv_filename
    PLOT_DURATION = END_TIME - START_TIME
    
    # 运行主函数
    main()