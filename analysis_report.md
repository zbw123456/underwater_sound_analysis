# Underwater Sound Analysis Report
## Baltic Sea Acoustic Clips - Doctoral Assignment

### Executive Summary
This report presents a comprehensive acoustic analysis of 6 distinct underwater sound clips extracted from a Baltic Sea recording. The analysis includes time-domain waveforms, spectrograms, broadband sound levels, one-third octave band analysis, and spectral characterization.

### Data Overview
**Source**: Baltic Sea acoustic recording (untitled.flac)
**Sampling Rate**: 16 kHz
**Total Duration**: 57.52 seconds
**Analysis Segment**: 20-27 seconds (main acoustic event)
**Extracted Clips**: 6 × 1-second segments
**Acoustic Reference**: 1 µPa (standard for underwater acoustics)

### Methodology

#### Signal Processing
All clips were extracted from the primary acoustic event window (20-27 seconds) where significant acoustic energy was detected. Each clip is exactly 1 second long for consistent comparison.

#### Acoustic Measurements

**1. Broadband Sound Level (dB re 1 µPa)**
Calculated as: L_B = 20 × log₁₀(RMS / 1 µPa)
- Represents total acoustic energy across all frequencies
- Standard metric for underwater noise characterization
- Used for comparing overall acoustic intensity between clips

**2. One-Third Octave Band Analysis**
- Standard frequency bands from 63 Hz to 10 kHz (ISO 266-1997)
- Bandpass filter: Butterworth 4th-order for precision
- Provides frequency-dependent energy distribution
- Reveals dominant acoustic features in specific frequency ranges

**3. Spectral Analysis**
- Welch power spectral density with 4096-point FFT
- Short-time Fourier transform spectrograms
- Logarithmic frequency scale for better visualization
- dB reference scale for power measurements

#### Feature Extraction
- **RMS Amplitude**: Root mean square voltage level (linear energy measure)
- **Peak Amplitude**: Maximum absolute value in signal
- **Crest Factor**: Peak to RMS ratio (measure of signal impulsivity)
- **Dominant Frequency**: Frequency with maximum power
- **Spectral Centroid**: Center of mass of frequency spectrum
- **Zero Crossing Rate**: Number of zero amplitude crossings per second

### Results Summary

#### Broadband Sound Levels
- Clip 1: 85.4 dB re 1 µPa
- Clip 2: 85.2 dB re 1 µPa
- Clip 3: 87.3 dB re 1 µPa
- Clip 4: 87.7 dB re 1 µPa
- Clip 5: 86.3 dB re 1 µPa
- Clip 6: 85.6 dB re 1 µPa

**Average Broadband Level**: 86.3 dB re 1 µPa
**Level Range**: 85.2 to 87.7 dB

#### Dominant Frequency Analysis
All clips show energy concentrated in low-frequency region:
- Clip 1: 53 Hz, Spectral Centroid: 102.7 Hz
- Clip 2: 54 Hz, Spectral Centroid: 97.0 Hz
- Clip 3: 53 Hz, Spectral Centroid: 79.6 Hz
- Clip 4: 53 Hz, Spectral Centroid: 78.9 Hz
- Clip 5: 53 Hz, Spectral Centroid: 86.8 Hz
- Clip 6: 53 Hz, Spectral Centroid: 90.0 Hz

### Acoustic Source Identification

#### Analysis

Based on the measured acoustic characteristics:

**Frequency Content**: 
- Energy predominantly in 50-100 Hz range
- Consistent spectral signature across all clips
- Indicates coherent acoustic source

**Temporal Characteristics**:
- Relatively uniform energy levels across clips
- Crest factors 2.8-3.3 indicate moderate impulsivity
- Suggests continuous or quasi-continuous source

**Acoustic Level**:
- Broadband levels 85-88 dB re 1 µPa are moderate
- Typical of marine vessel noise or large machinery
- Significantly above ambient underwater noise

#### Potential Source Categories

**Low-Frequency Machinery (Primary)**
- Ship propulsion systems and diesel engines
- Gearbox noise
- Large industrial equipment
- Characteristic low-frequency energy concentrated below 100 Hz

**Marine Biology (Secondary)**
- Large whale calls (typically 10-200 Hz)
- Fish vocalizations with low-frequency components
- Group activities or migration patterns

**Environmental Factors**
- Baltic Sea shipping traffic
- Industrial marine activities
- Harbor operations (20-28 second window likely correlates with nearby activity)

### Interpretation of Results

#### Spectral Patterns
The consistent dominant frequency near 53 Hz across all clips indicates:
1. Stable acoustic source (not rapidly varying)
2. Likely mechanical in origin (characteristic machinery signature)
3. Possible rotating machinery (turbine, propeller blade passing frequency patterns)

#### Temporal Consistency
Nearly identical RMS levels (0.0187-0.0244 V) across sequential 1-second clips suggests:
1. Continuous acoustic event lasting 7+ seconds
2. No significant fluctuations in source intensity
3. Stable operating conditions of source

#### Broadband Level Interpretation
Levels of 85-88 dB re 1 µPa place this sound in the moderate range:
- Comparable to large commercial vessel noise
- Higher than ambient ocean noise (~75-80 dB)
- Lower than major industrial operations (>100 dB)

### Analysis Outputs

**Visualizations Generated:**
1. `01_waveforms.png` - Time-domain waveforms of all 6 clips
   - Shows amplitude variations over 1-second segments
   - Reveals impulsive vs. continuous character
   
2. `02_spectrograms.png` - Spectrograms showing time-frequency content
   - Logarithmic frequency scale highlights low-frequency content
   - dB scale reveals energy distribution
   - Individual clips allow tracking of spectral evolution
   
3. `03_broadband_levels.png` - Bar chart of overall sound levels
   - Comparison of total acoustic energy
   - Shows variation between clips
   
4. `04_octave_bands.png` - Frequency distribution across standard bands
   - Reveals which frequency ranges dominate
   - Shows energy at 63 Hz and 79 Hz bands
   - Useful for source identification
   
5. `05_frequency_spectra.png` - Power spectral density plots
   - Fine-resolution frequency analysis
   - Shows spectral peaks clearly
   - Reveals harmonic structure

**Processing Parameters:**
- FFT Window: Hann window, 2048-point FFT
- Hop Length: 512 samples (32 ms)
- Mel-scale spectrogram: 128 frequency bands
- Octave Bands: 1/3 octave spacing (ISO 266-1997)
- Welch PSD: 4096-point FFT, 50% overlap

### Underwater Acoustics Context

Underwater sound propagation and measurement differ significantly from terrestrial acoustics:

**Key Differences**:
- Sound speed in water (~1500 m/s) vs air (~343 m/s)
- Greater transmission distances (up to 100+ km for low frequencies)
- Different absorption characteristics (frequency-dependent)
- Multipath propagation due to surface and bottom reflection
- Marine biological noise background

**Reference Standards**:
- International standard reference: 1 µPa (unlike air which uses 20 µPa)
- Hydrophone sensitivity typically ≈ -165 dB re 1 V/µPa
- Background noise in Baltic Sea typically 75-85 dB

### Conclusions

The acoustic analysis reveals detailed characteristics of Baltic Sea underwater sounds. The extracted clips represent distinct time samples from a coherent acoustic event, allowing for detailed spectral and temporal analysis. The measured sound levels and frequency content provide insights into the acoustic environment and potential source mechanisms.

**Key Findings**:
1. Single dominant acoustic source lasting 7+ seconds
2. Characteristic low-frequency signature (~53 Hz dominant)
3. Moderate sound level consistent with vessel or machinery noise
4. Relatively stable source with minimal fluctuations
5. Frequency content consistent with propulsion machinery or similar systems

**Applications of Results**:
- Marine acoustic baseline characterization
- Underwater noise monitoring and assessment
- Shipping traffic impact evaluation
- Marine ecosystem assessment (noise effect on marine life)
- Acoustic communication system design
- Environmental impact monitoring

### Technical Notes

**Software Used**: Python with scientific libraries
- NumPy: Numerical computations
- SciPy: Digital signal processing (filtering, FFT)
- Librosa: Audio analysis and spectrogram generation
- Matplotlib: Visualization and plotting
- Soundfile: FLAC audio file reading

**Signal Processing Methods**:
- Butterworth filtering (4th-order) for octave band analysis
- Welch method for power spectral density estimation
- Short-time Fourier transform (STFT) for spectrograms
- Standard acoustic metrics (RMS, crest factor, spectral centroid)

**Reproducibility**:
All analysis parameters are documented and can be adjusted for different analysis requirements. The Python code is provided for full reproducibility and modification.

---
**Analysis Date**: 2026-03-22
**Assignment**: Doctoral Candidate - Underwater Acoustic Analysis
**Methodology**: Digital signal processing with standard underwater acoustic metrics
**Duration**: 6 × 1-second clips from 57.52-second recording
**Audio File**: untitled.flac (Baltic Sea recording)
