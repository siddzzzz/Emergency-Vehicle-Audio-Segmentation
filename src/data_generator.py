import os
import math
import numpy as np
import scipy.io.wavfile as wavfile
from pathlib import Path

# Relative base directory for dataset
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
SIREN_DIR = DATA_DIR / "sirens"
TRAFFIC_DIR = DATA_DIR / "traffic"


def generate_siren_wail(duration_sec=4.0, sample_rate=16000):
    """
    Synthesize an Ambulance/Police Wail Siren:
    Slow frequency modulation between ~600 Hz and ~1200 Hz.
    """
    t = np.linspace(0, duration_sec, int(sample_rate * duration_sec), endpoint=False)
    # Modulation frequency 0.25 Hz (period of 4 seconds)
    mod_freq = 0.25
    f_min, f_max = 600.0, 1300.0
    
    # Triangle wave frequency modulation
    phase = 2 * np.pi * (f_min * t + (f_max - f_min) * (0.5 * (t * mod_freq - np.floor(t * mod_freq + 0.5))))
    # Sine modulation curve
    freq_inst = f_min + (f_max - f_min) * 0.5 * (1 + np.sin(2 * np.pi * mod_freq * t - np.pi / 2))
    phase = 2 * np.pi * np.cumsum(freq_inst) / sample_rate
    
    signal = 0.7 * np.sin(phase) + 0.2 * np.sin(2 * phase)  # Harmonics
    return (signal / np.max(np.abs(signal))).astype(np.float32)


def generate_siren_yelp(duration_sec=4.0, sample_rate=16000):
    """
    Synthesize a Yelp Siren:
    Fast frequency modulation between ~600 Hz and ~1400 Hz.
    """
    t = np.linspace(0, duration_sec, int(sample_rate * duration_sec), endpoint=False)
    # Modulation frequency 2.5 Hz (period of 0.4 seconds)
    mod_freq = 2.5
    f_min, f_max = 650.0, 1450.0
    
    freq_inst = f_min + (f_max - f_min) * 0.5 * (1 + np.sin(2 * np.pi * mod_freq * t))
    phase = 2 * np.pi * np.cumsum(freq_inst) / sample_rate
    
    signal = 0.8 * np.sin(phase) + 0.15 * np.sin(2 * phase)
    return (signal / np.max(np.abs(signal))).astype(np.float32)


def generate_siren_hilo(duration_sec=4.0, sample_rate=16000):
    """
    Synthesize a European / Firetruck Hi-Lo Siren:
    Alternating high (900 Hz) and low (700 Hz) tones every 0.5s.
    """
    t = np.linspace(0, duration_sec, int(sample_rate * duration_sec), endpoint=False)
    period = 1.0  # 0.5s high, 0.5s low
    freq = np.where((t % period) < 0.5, 900.0, 700.0)
    phase = 2 * np.pi * np.cumsum(freq) / sample_rate
    
    signal = 0.75 * np.sin(phase) + 0.2 * np.sin(3 * phase)
    return (signal / np.max(np.abs(signal))).astype(np.float32)


def generate_traffic_noise(duration_sec=4.0, sample_rate=16000, noise_type="rumble"):
    """
    Synthesize background traffic noise:
    - Pink noise base (engine rumble)
    - Random amplitude modulation (passing cars)
    - Occasional car horn bursts
    """
    num_samples = int(sample_rate * duration_sec)
    t = np.linspace(0, duration_sec, num_samples, endpoint=False)
    
    # Generate pink noise (1/f noise for heavy low frequency engine rumble)
    unequal_noise = np.random.randn(num_samples)
    fft_noise = np.fft.rfft(unequal_noise)
    frequencies = np.fft.rfftfreq(num_samples, 1 / sample_rate)
    frequencies[0] = 1.0  # avoid division by zero
    pink_filter = 1 / np.sqrt(frequencies)
    pink_filter /= np.max(pink_filter)
    
    pink_fft = fft_noise * pink_filter
    pink_noise = np.fft.irfft(pink_fft, n=num_samples)
    
    # Engine revs modulation
    engine_mod = 0.5 + 0.5 * np.sin(2 * np.pi * 0.3 * t + np.random.rand() * 2 * np.pi)
    traffic_sound = pink_noise * engine_mod
    
    if noise_type == "horns":
        # Add random car horn honk
        horn_start = int(0.3 * num_samples)
        horn_len = int(0.4 * sample_rate)
        if horn_start + horn_len < num_samples:
            t_horn = t[horn_start:horn_start + horn_len]
            horn = 0.3 * (np.sin(2 * np.pi * 420 * t_horn) + np.sin(2 * np.pi * 510 * t_horn))
            traffic_sound[horn_start:horn_start + horn_len] += horn

    traffic_sound = traffic_sound / (np.max(np.abs(traffic_sound)) + 1e-6)
    return traffic_sound.astype(np.float32)


def generate_sample_dataset(num_samples=10, sample_rate=16000, duration_sec=4.0):
    """
    Generates sample audio files in data/sirens and data/traffic for standalone testing.
    Relative paths are used throughout.
    """
    SIREN_DIR.mkdir(parents=True, exist_ok=True)
    TRAFFIC_DIR.mkdir(parents=True, exist_ok=True)
    
    siren_generators = [
        ("wail", generate_siren_wail),
        ("yelp", generate_siren_yelp),
        ("hilo", generate_siren_hilo),
    ]
    
    print(f"Generating synthetic sirens in: {SIREN_DIR}")
    for i in range(num_samples):
        s_name, s_gen = siren_generators[i % len(siren_generators)]
        s_audio = s_gen(duration_sec=duration_sec, sample_rate=sample_rate)
        file_path = SIREN_DIR / f"siren_{s_name}_{i+1:02d}.wav"
        # Convert float32 [-1, 1] to int16 for standard WAV format
        wavfile.write(str(file_path), sample_rate, (s_audio * 32767).astype(np.int16))
    
    print(f"Generating synthetic traffic noise in: {TRAFFIC_DIR}")
    for i in range(num_samples):
        n_type = "horns" if i % 2 == 0 else "rumble"
        t_audio = generate_traffic_noise(duration_sec=duration_sec, sample_rate=sample_rate, noise_type=n_type)
        file_path = TRAFFIC_DIR / f"traffic_{n_type}_{i+1:02d}.wav"
        wavfile.write(str(file_path), sample_rate, (t_audio * 32767).astype(np.int16))
        
    print("Sample dataset generation complete!")


if __name__ == "__main__":
    generate_sample_dataset(num_samples=12)
