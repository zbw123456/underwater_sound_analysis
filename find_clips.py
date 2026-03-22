"""
Find and extract 6 distinct underwater sound clips
"""

import soundfile as sf
import numpy as np
import librosa
from scipy import signal

# Load audio
data, sr = sf.read('untitled.flac')
print(f"Audio: {len(data)} samples at {sr} Hz ({len(data)/sr:.2f} seconds)")

# Method 1: Energy envelope with adaptive threshold
hop_length = 512
S = librosa.feature.melspectrogram(y=data, sr=sr, hop_length=hop_length, n_mels=64)
energy = np.sqrt(np.sum(S**2, axis=0))
energy_normalized = (energy - np.min(energy)) / (np.max(energy) - np.min(energy) + 1e-10)

print(f"Energy levels: min={energy.min():.2e}, max={energy.max():.2e}, mean={energy.mean():.2e}")

# Find all significant peaks
peaks, properties = signal.find_peaks(energy_normalized, 
                                       distance=1000,  # At least 0.5s apart
                                       height=0.3,      # Minimum height 30%
                                       prominence=0.1)  # Minimum prominence

print(f"\nFound {len(peaks)} peaks above threshold")
if len(peaks) > 0:
    print(f"Peak times (seconds): {peaks * hop_length / sr}")
    print(f"Peak heights: {energy_normalized[peaks]}")

# If not enough peaks, lower threshold
if len(peaks) < 6:
    print(f"\nNot enough peaks, lowering threshold...")
    peaks, properties = signal.find_peaks(energy_normalized, 
                                          distance=800,
                                          height=0.2,
                                          prominence=0.05)
    print(f"Found {len(peaks)} peaks")
    if len(peaks) > 0:
        print(f"Peak times (seconds): {peaks * hop_length / sr}")

# Get the highest energy 6 peaks (or use more if available)
if len(peaks) >= 6:
    peak_heights = energy_normalized[peaks]
    top_indices = np.argsort(peak_heights)[-6:]
    selected_peaks = peaks[sorted(top_indices)]
    print(f"\nSelected top 6 peaks at indices: {selected_peaks}")
    print(f"Selected peaks (seconds): {selected_peaks * hop_length / sr}")
elif len(peaks) > 0:
    selected_peaks = peaks
    print(f"\nUsing all {len(peaks)} detected peaks")
else:
    # Fallback: divide audio into 6 equal segments
    print("\nNo peaks found, using uniform division")
    clip_duration_samples = sr * 1  # 1 second
    segment_centers = np.linspace(clip_duration_samples//2, 
                                  len(data) - clip_duration_samples//2,
                                  6)
    selected_peaks = (segment_centers / hop_length).astype(int)
    print(f"Uniform centers (seconds): {selected_peaks * hop_length / sr}")

# Extract clips centered on detected peaks
clip_duration = sr  # 1 second
clips = []
clip_times = []

for peak_idx in selected_peaks:
    center_sample = min(peak_idx * hop_length, len(data) - 1)
    start = max(0, center_sample - clip_duration // 2)
    end = min(len(data), start + clip_duration)
    
    if end - start < clip_duration:
        start = max(0, end - clip_duration)
    
    clip = data[start:end]
    if len(clip) < clip_duration:
        clip = np.pad(clip, (0, clip_duration - len(clip)), mode='constant')
    
    clips.append(clip)
    clip_times.append(center_sample / sr)
    
    rms = np.sqrt(np.mean(clip**2))
    print(f"\nClip at {center_sample/sr:.2f}s: RMS={rms:.6f}, samples={len(clip)}")

print(f"\n\nExtracted {len(clips)} clips")
print(f"Clip times (seconds): {clip_times}")

# Save clips to file for reference
with open('detected_clips.txt', 'w') as f:
    f.write("Detected Clip Centers (seconds):\n")
    for i, t in enumerate(clip_times):
        f.write(f"Clip {i+1}: {t:.2f}s\n")

print("\nSaved clip information to detected_clips.txt")
