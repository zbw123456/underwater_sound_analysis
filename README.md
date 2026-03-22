Underwater Sound Analysis - Baltic Sea Recording

## Project Overview
This is a complete acoustic analysis of 6 underwater sound clips extracted from a Baltic Sea recording (untitled.flac). The analysis includes broadband sound levels, 1/3 octave band frequency analysis, spectrograms, and comprehensive feature extraction.

## Files Included

### Analysis Code
- **`underwater_acoustic_analysis.py`** - Main analysis script (fully runnable and self-contained)
  - Load and segment the audio file
  - Calculate broadband sound levels
  - Perform 1/3 octave band analysis
  - Extract acoustic features
  - Generate all visualizations
  - Create comprehensive analysis report

### Audio Data
- **`untitled.flac`** - Original Baltic Sea acoustic recording
  - Sampling rate: 16 kHz
  - Duration: 57.52 seconds
  - Format: Free Lossless Audio Codec (FLAC)

### Analysis Results

#### Visualizations
1. **`01_waveforms.png`** - Time-domain waveforms of all 6 clips
   - Shows amplitude variations for each 1-second segment
   - Reveals signal characteristics and dynamics

2. **`02_spectrograms.png`** - Spectrograms with time-frequency representation
   - Logarithmic frequency scale (better for acoustic analysis)
   - dB scale power representation
   - Shows spectral evolution over time

3. **`03_broadband_levels.png`** - Broadband sound level comparison
   - Bar chart of overall acoustic energy
   - Measured in dB re 1 µPa (standard underwater reference)
   - Clips 1-6: 85.2-87.7 dB range

4. **`04_octave_bands.png`** - 1/3 octave band frequency analysis
   - Standard ISO 266-1997 frequency bands (63 Hz - 10 kHz)
   - Shows energy distribution across frequency ranges
   - Reveals dominant frequency characteristics

5. **`05_frequency_spectra.png`** - Power Spectral Density plots
   - Welch method with 4096-point FFT
   - Fine-resolution frequency analysis
   - Shows spectral peaks and harmonics

#### Documentation
- **`analysis_report.md`** - Comprehensive analysis report
  - Methodology and signal processing details
  - Results summary with all measurements
  - Acoustic source identification and interpretation
  - Technical notes on processing parameters
  - Guidelines for interpreting results

## Quick Start

### Requirements
```bash
pip install soundfile numpy scipy librosa matplotlib
```

### Running the Analysis
```bash
python underwater_acoustic_analysis.py
```

The script will:
1. Load the FLAC audio file
2. Extract 6 one-second clips from the main acoustic event (20-28 seconds)
3. Calculate acoustic measurements for each clip:
   - Broadband sound levels (dB re 1 µPa)
   - 1/3 octave band levels
   - Spectral features (dominant frequency, spectral centroid, etc.)
4. Generate 5 visualization PNG files
5. Create a detailed analysis report (Markdown format)
6. Print analysis results to console

**Estimated runtime**: 10-15 seconds

## Key Findings

### Acoustic Measurements
- **Broadband Levels**: 85.2 - 87.7 dB re 1 µPa (average: 86.3 dB)
- **Dominant Frequency**: ~53 Hz (consistent across all clips)
- **Frequency Range**: 50-100 Hz concentrated energy
- **Spectral Centroid**: 78.9 - 102.7 Hz

### Source Characteristics
- Single coherent acoustic event lasting 7+ seconds
- Low-frequency machinery signature
- Possible origins: vessel propulsion, industrial equipment, or marine machinery
- Moderate sound level typical of marine vessel noise

### Acoustic Profile
- Impulsivity: Crest factors 2.84-3.30 (moderate)
- Continuity: Relatively stable energy across clips
- Spectrum: Strong concentration in low frequencies
- Environment: Baltic Sea marine acoustic environment

## Technical Details

### Analysis Methodology
1. **Signal Processing**
   - Audio loaded from FLAC file at 16 kHz sampling rate
   - Clips extracted from time window 20-28 seconds
   - Each clip: exactly 1 second (16,000 samples)

2. **Acoustic Calculations**
   - **Broadband Level**: L_B = 20 × log₁₀(RMS / 1 µPa)
   - **Octave Bands**: Butterworth 4th-order bandpass filters
   - **Reference**: 1 µPa (standard for underwater acoustics)

3. **Feature Extraction**
   - RMS amplitude (energy level)
   - Peak amplitude (maximum value)
   - Crest factor (peak/RMS ratio, impulsivity)
   - Dominant frequency (peak in spectrum)
   - Spectral centroid (center of mass)
   - Zero crossing rate (temporal characteristics)

4. **Visualization Methods**
   - STFT spectrograms with Hann window (2048-point FFT)
   - Welch power spectral density (4096-point FFT)
   - Logarithmic frequency scales (better for acoustics)
   - dB power scales (standard reference)

### Processing Parameters
- Spectrogram: Hann window, 2048-point FFT, hop length 512
- PSD: Welch method, 4096-point FFT, 50% overlap
- Octave Bands: 1/3 octave spacing per ISO 266-1997
- Frequency Range: 20 Hz to 8 kHz (covers relevant underwater acoustic range)

## Interpretation Guide

### Sound Level Categories (dB re 1 µPa)
- **> 150 dB**: Extremely high (large vessels, heavy industry)
- **130-150 dB**: High energy (shipping lanes, intense activity)
- **110-130 dB**: Moderate (typical marine environment)
- **100-110 dB**: Low-moderate (ambient underwater noise)
- **< 100 dB**: Low energy (quiet regions)

### Spectral Patterns
- **Narrow-band**: Tonal sources (machinery, biological)
- **Broadband**: Transient sources (cavitation, breaking waves)
- **Low-freq dominant**: Large machinery, distant sources
- **High-freq dominant**: Biological activity, small turbulence

## Acoustic Background

### Underwater Sound Characteristics
- Sound speed in water: ~1500 m/s (vs. 343 m/s in air)
- Greater transmission distance: up to 100+ km for low frequencies
- Complex propagation: surface and bottom reflections
- Frequency-dependent absorption

### Reference Standards
- Underwater hydrophone reference: 1 µPa (vs. 20 µPa in air)
- Baltic Sea background noise: typically 75-85 dB
- Typical marine vessel noise: 120-160 dB depending on size/speed

## Usage Notes

### For Researchers
- All acoustic calculations use standard underwater reference (1 µPa)
- Analysis follows ISO and international acoustic standards
- Results comparable with other underwater monitoring systems
- Can be adapted for different audio files with similar properties

### For Modification
- Edit clip start times in line ~55-56 to analyze different regions
- Adjust frequency band limits (line ~135) for specific ranges
- Change FFT parameters in visualization functions
- Modify visualization styles in matplotlib sections

### File Management
- All PNG files (~1.5 MB total) are publication-quality
- Report is in standard Markdown format (editable in any text editor)
- Python script is standalone (no configuration files needed)
- All dependencies installed via pip

## Dependencies

### Python Libraries
```
soundfile          # FLAC audio file support
numpy              # Numerical computations
scipy              # Signal processing
librosa            # Audio analysis library
matplotlib         # Visualization
```

### System Requirements
- Python 3.8+
- ~50 MB disk space for visualizations
- ~1 GB RAM (typically uses <100 MB)

## Questions & Notes

### About the Analysis
- **Why 6 clips?** Assignment requirement for analyzing distinct segments
- **Why 20-28 seconds?** Primary acoustic event detected in this window
- **Why 1 second each?** Standard duration for acoustic analysis
- **Why 1 µPa reference?** Standard for underwater acoustics (ISO/IEC 60027-3)

### About the Results
- **Why these frequencies?** Results from signal content, not predetermined
- **Why consistent across clips?** Indicates continuous, stable acoustic source
- **What caused the sound?** Analysis reveals characteristics but not source identity with certainty
- **How does this compare?** Levels are typical for marine vessel noise

## Contact & Credits

This analysis was performed as part of a doctoral assignment on underwater acoustic analysis of Baltic Sea recordings.

**Analysis Date**: 2026-03-22
**Software**: Python (NumPy, SciPy, Librosa, Matplotlib)
**Methods**: Digital signal processing with standard acoustic metrics

---

For questions about the analysis methods or results, refer to the comprehensive analysis_report.md file included in this directory.
