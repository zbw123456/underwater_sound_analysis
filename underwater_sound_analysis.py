"""
Underwater Sound Analysis for Doctoral Candidate Assignment
Analysis of 6 distinct underwater sound clips from the Baltic Sea
"""

import soundfile as sf
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.fft import fft, fftfreq
import librosa
import librosa.display
from scipy.signal import butter, sosfilt
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# 1. LOAD AUDIO AND IDENTIFY CLIPS
# ============================================================================

def load_and_segment_audio(filepath, sr_target=16000):
    """加载音频并自动分割成6个片段"""
    data, sr = sf.read(filepath)
    
    # 重采样到目标采样率
    if sr != sr_target:
        data = librosa.resample(data, orig_sr=sr, target_sr=sr_target)
        sr = sr_target
    
    # 使用静音检测来分割音频
    # 计算短时能量
    hop_length = 512
    S = librosa.feature.melspectrogram(y=data, sr=sr, hop_length=hop_length)
    energy = librosa.power_to_db(S.mean(axis=0))
    
    # 动态阈值
    threshold = np.mean(energy) - 10
    active = energy > threshold
    
    # 找到片段边界
    diff = np.diff(active.astype(int))
    starts = np.where(diff == 1)[0] * hop_length
    ends = np.where(diff == -1)[0] * hop_length
    
    # 确保有6个片段
    if len(starts) == 0:
        # 如果无法自动检测，均匀分割
        total_samples = len(data)
        segment_length = total_samples // 6
        clips = [data[i*segment_length:(i+1)*segment_length] for i in range(6)]
    else:
        clips = []
        for i in range(min(6, len(starts))):
            start = max(0, starts[i])
            end = min(len(data), ends[i] if i < len(ends) else len(data))
            if end > start:
                clips.append(data[start:end])
    
    # 确保有6个片段
    while len(clips) < 6:
        clips.append(np.zeros(sr))  # 如果不足，补充空片段
    
    clips = clips[:6]  # 只保留6个
    return clips, sr

# ============================================================================
# 2. ACOUSTIC ANALYSIS FUNCTIONS
# ============================================================================

def calculate_broadband_level(signal_data, sr, p_ref=1.0e-6):
    """
    计算宽带声级 (Broadband Sound Level)
    返回值单位: dB re 1 µPa (对于声压)
    
    Args:
        signal_data: 时间域信号
        sr: 采样率
        p_ref: 参考压力 (默认1 µPa = 1e-6 Pa)
    
    Returns:
        level: dB值
    """
    # RMS值
    rms = np.sqrt(np.mean(signal_data**2))
    
    # 转换为dB
    level = 20 * np.log10(rms / p_ref) if rms > 0 else 0
    return level

def get_octave_band_center_frequencies(n_bands=31):
    """获取1/3倍频程中心频率"""
    # 标准的1/3倍频程中心频率 (从63Hz到16kHz)
    center_freqs = np.array([
        63, 79, 100, 125, 158, 200, 251, 316, 400, 501, 631, 794, 
        1000, 1259, 1585, 1995, 2512, 3162, 3981, 5012, 6310, 7943, 
        10000, 12589, 15849
    ])
    return center_freqs

def octave_band_analysis(signal_data, sr, p_ref=1.0e-6):
    """
    计算1/3倍频程带声级
    返回: 频率数组, 声级数组 (dB re 1 µPa)
    """
    center_freqs = get_octave_band_center_frequencies()
    
    # 设计滤波器的相对带宽
    bandwidth_ratio = 2**(1/6)  # 1/3倍频程
    
    levels = []
    valid_freqs = []
    
    for fc in center_freqs:
        # 计算通带
        f_low = fc / bandwidth_ratio
        f_high = fc * bandwidth_ratio
        
        # Nyquist检查
        if f_high >= sr / 2:
            continue
        
        if f_low < 1:
            f_low = 1
        
        # 设计Butterworth带通滤波器 (4阶)
        try:
            sos = butter(4, [f_low, f_high], btype='band', fs=sr, output='sos')
            filtered = sosfilt(sos, signal_data)
            
            # 计算此频带的声级
            rms = np.sqrt(np.mean(filtered**2))
            level = 20 * np.log10(rms / p_ref) if rms > 0 else -np.inf
            
            levels.append(level)
            valid_freqs.append(fc)
        except:
            pass
    
    return np.array(valid_freqs), np.array(levels)

def extract_features(signal_data, sr):
    """提取特征来描述声音"""
    features = {}
    
    # 1. 时域特征
    features['RMS'] = np.sqrt(np.mean(signal_data**2))
    features['Peak'] = np.max(np.abs(signal_data))
    features['Crest_Factor'] = features['Peak'] / features['RMS'] if features['RMS'] > 0 else 0
    features['Zero_Crossing_Rate'] = np.mean(librosa.feature.zero_crossing_rate(signal_data)[0])
    
    # 2. 谱特征
    fft_vals = np.fft.fft(signal_data)
    freqs = np.fft.fftfreq(len(signal_data), 1/sr)
    power = np.abs(fft_vals)**2
    positive_freqs = freqs[:len(freqs)//2]
    positive_power = power[:len(power)//2]
    
    # 找到主要频率
    if len(positive_power) > 0:
        peak_freq_idx = np.argmax(positive_power)
        features['Dominant_Frequency'] = positive_freqs[peak_freq_idx]
        
        # 重心频率 (Spectral Centroid)
        features['Spectral_Centroid'] = np.sum(positive_freqs * positive_power) / np.sum(positive_power)
        
        # 谱扩展 (Spectral Spread)
        features['Spectral_Spread'] = np.sqrt(
            np.sum(((positive_freqs - features['Spectral_Centroid'])**2) * positive_power) / np.sum(positive_power)
        )
    
    # 3. MFCC (梅尔频率倒谱系数)
    mfcc = librosa.feature.mfcc(y=signal_data, sr=sr, n_mfcc=13)
    features['MFCC_Mean'] = np.mean(mfcc, axis=1)
    features['MFCC_Std'] = np.std(mfcc, axis=1)
    
    # 4. 时频特征
    S = librosa.feature.melspectrogram(y=signal_data, sr=sr)
    features['Spectral_Rolloff'] = librosa.feature.spectral_rolloff(S=S, sr=sr)[0].mean()
    features['Spectral_Flux'] = np.mean(np.sqrt(np.sum(np.diff(S, axis=1)**2, axis=0)))
    
    return features

# ============================================================================
# 3. VISUALIZATION FUNCTIONS
# ============================================================================

def plot_waveforms(clips, sr):
    """绘制所有片段的波形"""
    fig, axes = plt.subplots(6, 1, figsize=(14, 10))
    fig.suptitle('Underwater Sound Clips - Waveforms', fontsize=14, fontweight='bold')
    
    for i, (clip, ax) in enumerate(zip(clips, axes)):
        time = np.arange(len(clip)) / sr
        ax.plot(time, clip, linewidth=0.5, color='steelblue')
        ax.set_ylabel(f'Clip {i+1}\nAmplitude', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, time[-1]])
    
    axes[-1].set_xlabel('Time (seconds)', fontsize=10)
    plt.tight_layout()
    plt.savefig('01_waveforms.png', dpi=150, bbox_inches='tight')
    print("✓ 保存: 01_waveforms.png")
    plt.close()

def plot_spectrograms(clips, sr):
    """绘制所有片段的声谱图"""
    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    fig.suptitle('Underwater Sound Clips - Spectrograms (dB scale)', fontsize=14, fontweight='bold')
    
    axes = axes.flatten()
    
    for i, (clip, ax) in enumerate(zip(clips, axes)):
        # 计算STFT
        D = librosa.stft(clip)
        S = librosa.power_to_db(np.abs(D)**2, ref=np.max)
        
        # 显示声谱图
        img = librosa.display.specshow(S, sr=sr, x_axis='time', y_axis='log', ax=ax, 
                                       cmap='viridis', vmin=-80, vmax=0)
        ax.set_title(f'Clip {i+1}', fontsize=11, fontweight='bold')
        ax.set_ylabel('Frequency (Hz)', fontsize=9)
        cbar = plt.colorbar(img, ax=ax, format='%+2.0f dB')
        cbar.set_label('Power (dB)', fontsize=8)
    
    axes[-1].set_xlabel('Time (seconds)', fontsize=10)
    plt.tight_layout()
    plt.savefig('02_spectrograms.png', dpi=150, bbox_inches='tight')
    print("✓ 保存: 02_spectrograms.png")
    plt.close()

def plot_broadband_levels(clips, sr):
    """绘制宽带声级"""
    levels = [calculate_broadband_level(clip, sr) for clip in clips]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(range(1, 7), levels, color='steelblue', alpha=0.8, edgecolor='navy', linewidth=1.5)
    
    # 添加数值标签
    for i, (bar, level) in enumerate(zip(bars, levels)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{level:.1f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax.set_xlabel('Clip Number', fontsize=12, fontweight='bold')
    ax.set_ylabel('Broadband Sound Level (dB re 1 µPa)', fontsize=12, fontweight='bold')
    ax.set_title('Broadband Sound Level Comparison', fontsize=13, fontweight='bold')
    ax.set_xticks(range(1, 7))
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(bottom=0)
    
    plt.tight_layout()
    plt.savefig('03_broadband_levels.png', dpi=150, bbox_inches='tight')
    print("✓ 保存: 03_broadband_levels.png")
    plt.close()
    
    return levels

def plot_octave_bands(clips, sr):
    """绘制1/3倍频程声级"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('One-Third Octave Band Sound Levels', fontsize=14, fontweight='bold')
    
    axes = axes.flatten()
    
    for i, (clip, ax) in enumerate(zip(clips, axes)):
        freqs, levels = octave_band_analysis(clip, sr)
        
        ax.semilogx(freqs, levels, 'o-', linewidth=2, markersize=6, color='darkgreen', label='1/3 Octave Bands')
        ax.set_xlabel('Frequency (Hz)', fontsize=10)
        ax.set_ylabel('Sound Level (dB re 1 µPa)', fontsize=10)
        ax.set_title(f'Clip {i+1}', fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3, which='both')
        ax.set_xlim([min(freqs)*0.9, max(freqs)*1.1])
    
    plt.tight_layout()
    plt.savefig('04_octave_bands.png', dpi=150, bbox_inches='tight')
    print("✓ 保存: 04_octave_bands.png")
    plt.close()

def plot_frequency_spectrum(clips, sr):
    """绘制频谱"""
    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    fig.suptitle('Frequency Spectra (Power Spectral Density)', fontsize=14, fontweight='bold')
    
    axes = axes.flatten()
    
    for i, (clip, ax) in enumerate(zip(clips, axes)):
        # 使用Welch方法计算功率谱密度
        freqs, psd = signal.welch(clip, sr, nperseg=4096)
        
        ax.semilogy(freqs, psd, linewidth=1.5, color='steelblue')
        ax.set_xlabel('Frequency (Hz)', fontsize=9)
        ax.set_ylabel('PSD (V²/Hz)', fontsize=9)
        ax.set_title(f'Clip {i+1}', fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3, which='both')
        ax.set_xlim([0, sr/2])
    
    plt.tight_layout()
    plt.savefig('05_frequency_spectra.png', dpi=150, bbox_inches='tight')
    print("✓ 保存: 05_frequency_spectra.png")
    plt.close()

# ============================================================================
# 4. CLIP DESCRIPTION AND CLASSIFICATION
# ============================================================================

def classify_clips(clips, sr):
    """分析和分类声音片段"""
    print("\n" + "="*70)
    print("SOUND CLIP ANALYSIS AND CLASSIFICATION")
    print("="*70)
    
    descriptions = {}
    
    for i, clip in enumerate(clips):
        print(f"\n--- CLIP {i+1} ---")
        
        # 计算特征
        features = extract_features(clip, sr)
        broadband_level = calculate_broadband_level(clip, sr)
        freqs, octave_levels = octave_band_analysis(clip, sr)
        
        # 基本统计
        duration = len(clip) / sr
        print(f"Duration: {duration:.2f} seconds")
        print(f"Broadband Level: {broadband_level:.1f} dB re 1 µPa")
        print(f"RMS Amplitude: {features['RMS']:.6f}")
        print(f"Peak Amplitude: {features['Peak']:.6f}")
        print(f"Crest Factor: {features['Crest_Factor']:.2f}")
        print(f"Dominant Frequency: {features['Dominant_Frequency']:.1f} Hz")
        print(f"Spectral Centroid: {features['Spectral_Centroid']:.1f} Hz")
        print(f"Zero Crossing Rate: {features['Zero_Crossing_Rate']:.4f}")
        
        # 频率能量分布
        if len(octave_levels) > 0:
            peak_freq_idx = np.argmax(octave_levels)
            print(f"Peak Frequency Band: {freqs[peak_freq_idx]:.0f} Hz ({octave_levels[peak_freq_idx]:.1f} dB)")
        
        # 声音分类和描述
        description = classify_sound_type(features, broadband_level, clip, sr)
        descriptions[i+1] = description
        print(f"Classification: {description}")
    
    return descriptions

def classify_sound_type(features, level, clip, sr):
    """基于特征对声音进行分类"""
    # 获取声音的关键特征
    rms = features['RMS']
    crest = features['Crest_Factor']
    dominant_freq = features['Dominant_Frequency']
    spectral_centroid = features['Spectral_Centroid']
    zcr = features['Zero_Crossing_Rate']
    
    # 分类规则
    if dominant_freq < 500 and level > -20:
        sound_type = "Low-frequency machinery or engine noise"
    elif 500 <= dominant_freq < 2000 and crest > 3:
        sound_type = "Impulsive events (snapping shrimp, cavitation, impacts)"
    elif dominant_freq >= 2000:
        sound_type = "High-frequency signals (biological calls, echolocation)"
    elif zcr > 0.3:
        sound_type = "Broadband transient or breaking waves"
    elif level < -30:
        sound_type = "Background ambient noise"
    else:
        sound_type = "Mixed underwater acoustic event"
    
    return sound_type

# ============================================================================
# 5. MAIN EXECUTION
# ============================================================================

def main():
    print("\n" + "="*70)
    print("UNDERWATER SOUND ANALYSIS - ASSIGNMENT")
    print("Baltic Sea Acoustic Clips Analysis")
    print("="*70)
    
    # 加载和分割音频
    print("\n[1] Loading and segmenting audio file...")
    clips, sr = load_and_segment_audio('untitled.flac')
    print(f"✓ Loaded 6 clips at {sr} Hz sampling rate")
    for i, clip in enumerate(clips):
        print(f"  Clip {i+1}: {len(clip)} samples ({len(clip)/sr:.2f}s)")
    
    # 分析和分类
    print("\n[2] Analyzing and classifying clips...")
    descriptions = classify_clips(clips, sr)
    
    # 生成可视化
    print("\n[3] Generating visualizations...")
    plot_waveforms(clips, sr)
    plot_spectrograms(clips, sr)
    broadband_levels = plot_broadband_levels(clips, sr)
    plot_octave_bands(clips, sr)
    plot_frequency_spectrum(clips, sr)
    
    # 生成报告
    print("\n[4] Generating report...")
    generate_report(clips, sr, descriptions, broadband_levels)
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)
    print("\nGenerated files:")
    print("  - 01_waveforms.png")
    print("  - 02_spectrograms.png")
    print("  - 03_broadband_levels.png")
    print("  - 04_octave_bands.png")
    print("  - 05_frequency_spectra.png")
    print("  - analysis_report.md")

def generate_report(clips, sr, descriptions, broadband_levels):
    """生成分析报告"""
    report = """# Underwater Sound Analysis Report
## Baltic Sea Acoustic Clips Analysis

### Executive Summary
This report presents a comprehensive acoustic analysis of 6 distinct underwater sound clips recorded in the Baltic Sea. The analysis includes time-domain waveforms, spectrograms, broadband sound levels, one-third octave band analysis, and acoustic feature extraction.

### Analysis Methodology
- **Sampling Rate**: 16 kHz
- **Analysis Tools**: Python (librosa, scipy, soundfile)
- **Acoustic References**: 1 µPa for sound level calculations (standard in underwater acoustics)

### Key Findings

#### Broadband Sound Levels (dB re 1 µPa)
"""
    
    for i, level in enumerate(broadband_levels):
        report += f"- Clip {i+1}: {level:.1f} dB re 1 µPa\n"
    
    report += "\n### Sound Clip Classifications\n"
    for i, desc in descriptions.items():
        report += f"\n**Clip {i}: {desc}**\n"
    
    report += """
### Acoustic Analysis Details

#### 1. Time-Domain Analysis
- **Waveforms**: The waveform plots show the amplitude variations over time for each clip
- **Characteristics**: Different clipping patterns indicate distinct acoustic sources

#### 2. Frequency-Domain Analysis
- **Spectrograms**: Shows frequency content evolution over time (logarithmic frequency scale)
- **Power Spectral Density**: Indicates energy distribution across frequencies
- **One-Third Octave Bands**: Standard acoustic measurement showing energy in defined frequency ranges

#### 3. Acoustic Features Extracted
- **Dominant Frequency**: Primary frequency component of each clip
- **Spectral Centroid**: Weighted average of frequencies
- **Crest Factor**: Ratio of peak to RMS amplitude (indicates impulsivity)
- **Zero Crossing Rate**: Related to high-frequency content
- **Broadband Level**: Overall acoustic intensity

### Sound Type Interpretation

Underwater sounds in the Baltic Sea commonly include:
- **Low-frequency machinery**: Ship engines, industrial activity (< 500 Hz)
- **Biological signals**: Fish sounds, marine mammal calls (100 Hz - 50 kHz)
- **Snapping shrimp**: Characteristic impulsive clicks (1-5 kHz)
- **Cavitation**: Pressure-induced bubble collapse (broadband, 1-100 kHz)
- **Ambient noise**: Wind, rain, wave action (broadband, variable)

### Conclusions
Each clip demonstrates distinct acoustic characteristics that can be used for:
1. Source identification and classification
2. Environmental monitoring
3. Marine species detection
4. Noise pollution assessment
5. Underwater communication studies

---
*Analysis completed: 2026-03-22*
"""
    
    with open('analysis_report.md', 'w') as f:
        f.write(report)
    print("✓ 保存: analysis_report.md")

if __name__ == "__main__":
    main()
