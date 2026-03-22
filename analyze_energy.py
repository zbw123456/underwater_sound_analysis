#!/usr/bin/env python3
import soundfile as sf
import numpy as np
from scipy import signal

# Load audio
data, sr = sf.read('untitled.flac')
print(f"Total samples: {len(data)}")
print(f"Sampling rate: {sr} Hz")
print(f"Duration: {len(data)/sr:.2f} seconds\n")

# Analyze amplitude distribution
print("Amplitude statistics:")
print(f"  Min: {np.min(data):.6f}")
print(f"  Max: {np.max(data):.6f}")
print(f"  Mean: {np.mean(data):.6f}")
print(f"  Std: {np.std(data):.6f}")
print(f"  RMS: {np.sqrt(np.mean(data**2)):.6f}")

# Find time segments with energy above threshold
rms_window = sr // 4  # 250ms window
frame_rms = []
for i in range(0, len(data) - rms_window, rms_window):
    frame_rms.append(np.sqrt(np.mean(data[i:i+rms_window]**2)))

frame_rms = np.array(frame_rms)
threshold = np.mean(frame_rms) + 2 * np.std(frame_rms)
print(f"\nFrame RMS - Min: {frame_rms.min():.6f}, Max: {frame_rms.max():.6f}, Mean: {frame_rms.mean():.6f}")
print(f"Threshold (mean + 2*std): {threshold:.6f}")

# Find active regions
active_frames = np.where(frame_rms > threshold)[0]
print(f"Active frames above threshold: {len(active_frames)} out of {len(frame_rms)}")

if len(active_frames) > 0:
    print(f"\nActive regions:")
    active_times = active_frames * rms_window / sr
    for i in range(min(20, len(active_times))):
        print(f"  Frame {active_frames[i]}: {active_times[i]:.2f}s, RMS={frame_rms[active_frames[i]]:.6f}")
    
    # Find gaps to segment
    gaps = np.diff(active_frames)
    gap_threshold = 10  # Gap of 10 frames (2.5 seconds)
    segment_starts = [active_frames[0]]
    
    for i in range(len(gaps)):
        if gaps[i] > gap_threshold:
            segment_starts.append(active_frames[i+1])
    
    print(f"\nDetected {len(segment_starts)} sound segments based on gaps")
    for i, start in enumerate(segment_starts):
        print(f"  Segment {i+1} starts at frame {start} ({start * rms_window / sr:.2f}s)")

# Alternative: divide the audio uniformly into 6 segments
print(f"\nUniform segmentation into 6 clips:")
segment_duration = len(data) // 6
for i in range(6):
    start = i * segment_duration
    end = (i+1) * segment_duration
    clip = data[start:end]
    rms = np.sqrt(np.mean(clip**2))
    print(f"  Clip {i+1}: {start/sr:.2f}s to {end/sr:.2f}s, RMS={rms:.6f}")
