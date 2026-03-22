import soundfile as sf
import numpy as np
from scipy import signal
import librosa

data, sr = sf.read('untitled.flac')

# 使用更好的分割方法
hop_length = 512
S = librosa.feature.melspectrogram(y=data, sr=sr, hop_length=hop_length)
energy_db = librosa.power_to_db(S.mean(axis=0))

# 标准化能量
energy_norm = (energy_db - np.min(energy_db)) / (np.max(energy_db) - np.min(energy_db))

# 找到能量峰
peaks, properties = signal.find_peaks(energy_norm, distance=5000, height=0.3)

print(f"检测到 {len(peaks)} 个能量峰")

# 提取6个最大的峰
if len(peaks) >= 6:
    peak_heights = energy_norm[peaks]
    top_6_idx = np.argsort(peak_heights)[-6:]
    top_6_peaks = peaks[top_6_idx]
    top_6_peaks = np.sort(top_6_peaks)
else:
    top_6_peaks = np.sort(peaks)

# 生成片段标记
clip_centers = top_6_peaks * hop_length / sr
print(f"\n选定的6个片段中心 (秒):")
for i, center in enumerate(clip_centers):
    print(f"  Clip {i+1}: {center:.2f}s")

# 提取6个片段，每个1秒长
clip_duration = sr  # 1秒
clips = []
for center in clip_centers:
    center_sample = int(center * sr)
    start = max(0, center_sample - clip_duration // 2)
    end = min(len(data), start + clip_duration)
    if end - start < clip_duration:
        start = max(0, end - clip_duration)
    clip = data[start:end]
    if len(clip) < clip_duration:
        clip = np.pad(clip, (0, clip_duration - len(clip)), mode='constant')
    clips.append(clip)

print(f"\n提取的6个片段:")
for i, clip in enumerate(clips):
    print(f"  Clip {i+1}: {len(clip)} samples ({len(clip)/sr:.2f}s)")
    rms = np.sqrt(np.mean(clip**2))
    print(f"    - RMS: {rms:.6f}")
