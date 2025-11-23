import os
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import argparse
from tqdm import tqdm

# ================= 配置参数 =================
DATA_DIR = "drink_process_data"
OUTPUT_DIR = "visualization_normal_bp"
PPG_FS = 125
PLOT_DURATION = 10  # 绘制前10秒的信号

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
    if signal_type == 'ppg':
        # PPG信号：带通滤波 + 反转（透射式）
        from scipy import signal
        b, a = signal.butter(4, [0.7, 8.0], btype='band', fs=PPG_FS)
        filtered = signal.filtfilt(b, a, signal_data.astype(float))
        filtered = -filtered  # 反转信号
        return filtered
    elif signal_type == 'ecg':
        # ECG信号：带通滤波
        from scipy import signal
        b, a = signal.butter(4, [0.5, 40], btype='band', fs=PPG_FS)
        filtered = signal.filtfilt(b, a, signal_data.astype(float))
        return filtered
    else:
        # BP信号：轻微平滑
        from scipy import signal
        return signal.medfilt(signal_data, kernel_size=3)

# ================= 计算统计信息 =================
def calculate_signal_stats(signal_data, signal_name):
    """计算信号的统计信息"""
    if len(signal_data) == 0:
        return f"{signal_name}: No data"
    
    stats = {
        'mean': np.mean(signal_data),
        'std': np.std(signal_data),
        'min': np.min(signal_data),
        'max': np.max(signal_data),
        'length': len(signal_data)
    }
    
    return f"{signal_name}: mean={stats['mean']:.2f}, std={stats['std']:.2f}, range=[{stats['min']:.2f}, {stats['max']:.2f}], len={stats['length']}"

# ================= 可视化单个文件 =================
def visualize_file(file_path, output_path):
    """可视化单个MAT文件的信号"""
    try:
        # 加载数据
        data = sio.loadmat(file_path)
        
        # 检测信号
        ppg_key, bp_key, ecg_key = detect_signals(data)
        
        # 提取信号数据
        signals = {}
        if ppg_key:
            signals['PPG'] = np.array(data[ppg_key]).flatten()
        if bp_key:
            signals['BP'] = np.array(data[bp_key]).flatten()
        if ecg_key:
            signals['ECG'] = np.array(data[ecg_key]).flatten()
        
        if not signals:
            print(f"  No valid signals found in {os.path.basename(file_path)}")
            return False
        
        # 确定最小长度
        min_length = min(len(signal) for signal in signals.values())
        plot_samples = min(min_length, PLOT_DURATION * PPG_FS)
        
        # 创建图形
        fig = plt.figure(figsize=(15, 10))
        gs = GridSpec(len(signals), 1, figure=fig)
        
        # 绘制每个信号
        stats_text = []
        for i, (signal_name, signal_data) in enumerate(signals.items()):
            # 预处理信号
            processed_data = preprocess_signal(signal_data[:plot_samples], signal_name.lower())
            
            # 创建子图
            ax = fig.add_subplot(gs[i])
            
            # 生成时间轴
            time_axis = np.arange(plot_samples) / PPG_FS
            
            # 绘制信号
            ax.plot(time_axis, processed_data, linewidth=1.0, 
                   color='blue' if signal_name == 'PPG' else 
                         'red' if signal_name == 'BP' else 'green')
            
            # 设置子图属性
            ax.set_title(f'{signal_name} Signal - {os.path.basename(file_path)}', fontsize=12, fontweight='bold')
            ax.set_ylabel('Amplitude', fontsize=10)
            ax.grid(True, alpha=0.3)
            ax.set_xlim(0, plot_samples / PPG_FS)
            
            # 计算统计信息
            stats_text.append(calculate_signal_stats(processed_data, signal_name))
        
        # 设置x轴标签（只在最后一个子图）
        ax.set_xlabel('Time (seconds)', fontsize=10)
        
        # 添加统计信息文本框
        stats_str = '\n'.join(stats_text)
        fig.text(0.02, 0.02, stats_str, fontsize=9, verticalalignment='bottom',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))
        
        # 调整布局并保存
        plt.tight_layout(rect=[0, 0.05, 1, 0.98])
        plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        
        return True
        
    except Exception as e:
        print(f"  Error processing {file_path}: {str(e)}")
        return False

# ================= 主函数 =================
def main():
    """主函数：处理所有正常血压患者的MAT文件"""
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
    
    # 可视化每个文件
    success_count = 0
    for filename in tqdm(normal_bp_files, desc="Visualizing files"):
        file_path = os.path.join(DATA_DIR, filename)
        output_filename = os.path.splitext(filename)[0] + '.png'
        output_path = os.path.join(OUTPUT_DIR, output_filename)
        
        if visualize_file(file_path, output_path):
            success_count += 1
    
    print(f"\nVisualization completed!")
    print(f"Successfully processed: {success_count}/{len(normal_bp_files)} files")
    print(f"Output directory: {OUTPUT_DIR}")

if __name__ == "__main__":
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='Visualize normal BP patient signals')
    parser.add_argument('--data_dir', type=str, default=DATA_DIR, help='Input data directory')
    parser.add_argument('--output_dir', type=str, default=OUTPUT_DIR, help='Output directory for PNG files')
    parser.add_argument('--duration', type=int, default=PLOT_DURATION, help='Plot duration in seconds')
    
    args = parser.parse_args()
    
    # 更新配置
    DATA_DIR = args.data_dir
    OUTPUT_DIR = args.output_dir
    PLOT_DURATION = args.duration
    
    # 运行主函数
    main()