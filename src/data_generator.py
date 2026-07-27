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


def apply_burst_envelope(signal, sample_rate, duration_sec, start_sec=None, end_sec=None):
    """
    Applies a smooth cosine fade-in/fade-out envelope so siren starts and stops in the middle of audio.
    """
    num_samples = len(signal)
    if start_sec is None:
        start_sec = 2.0
    if end_sec is None:
        end_sec = min(duration_sec - 1.0, start_sec + 4.0)

    start_idx = int(start_sec * sample_rate)
    end_idx = int(end_sec * sample_rate)
    fade_len = int(0.3 * sample_rate)  # 300ms fade duration

    envelope = np.zeros(num_samples, dtype=np.float32)
    active_len = max(0, end_idx - start_idx)

    if active_len > 2 * fade_len:
        # Cosine fade-in
        fade_in = 0.5 * (1.0 - np.cos(np.pi * np.linspace(0, 1, fade_len)))
        # Cosine fade-out
        fade_out = 0.5 * (1.0 + np.cos(np.pi * np.linspace(0, 1, fade_len)))
        
        envelope[start_idx:start_idx + fade_len] = fade_in
        envelope[start_idx + fade_len:end_idx - fade_len] = 1.0
        envelope[end_idx - fade_len:end_idx] = fade_out
    else:
        envelope[start_idx:end_idx] = 1.0

    return (signal * envelope).astype(np.float32)


def generate_siren_wail(duration_sec=10.0, sample_rate=16000, start_sec=None, end_sec=None):
    """
    Synthesize an Ambulance/Police Wail Siren:
    Frequency modulation between ~600 Hz and ~1300 Hz.
    """
    num_samples = int(sample_rate * duration_sec)
    t = np.linspace(0, duration_sec, num_samples, endpoint=False)
    mod_freq = 0.3
    f_min, f_max = 600.0, 1300.0
    
    freq_inst = f_min + (f_max - f_min) * 0.5 * (1 + np.sin(2 * np.pi * mod_freq * t - np.pi / 2))
    phase = 2 * np.pi * np.cumsum(freq_inst) / sample_rate
    
    signal = 0.7 * np.sin(phase) + 0.2 * np.sin(2 * phase)
    signal = (signal / np.max(np.abs(signal))).astype(np.float32)

    if start_sec is not None or end_sec is not None or duration_sec > 5.0:
        signal = apply_burst_envelope(signal, sample_rate, duration_sec, start_sec, end_sec)

    return signal


def generate_siren_yelp(duration_sec=10.0, sample_rate=16000, start_sec=None, end_sec=None):
    """
    Synthesize a Yelp Siren:
    Fast frequency modulation between ~650 Hz and ~1450 Hz.
    """
    num_samples = int(sample_rate * duration_sec)
    t = np.linspace(0, duration_sec, num_samples, endpoint=False)
    mod_freq = 2.5
    f_min, f_max = 650.0, 1450.0
    
    freq_inst = f_min + (f_max - f_min) * 0.5 * (1 + np.sin(2 * np.pi * mod_freq * t))
    phase = 2 * np.pi * np.cumsum(freq_inst) / sample_rate
    
    signal = 0.8 * np.sin(phase) + 0.15 * np.sin(2 * phase)
    signal = (signal / np.max(np.abs(signal))).astype(np.float32)

    if start_sec is not None or end_sec is not None or duration_sec > 5.0:
        signal = apply_burst_envelope(signal, sample_rate, duration_sec, start_sec, end_sec)

    return signal


def generate_siren_hilo(duration_sec=10.0, sample_rate=16000, start_sec=None, end_sec=None):
    """
    Synthesize a European / Firetruck Hi-Lo Siren:
    Alternating high (900 Hz) and low (700 Hz) tones.
    """
    num_samples = int(sample_rate * duration_sec)
    t = np.linspace(0, duration_sec, num_samples, endpoint=False)
    period = 1.0
    freq = np.where((t % period) < 0.5, 900.0, 700.0)
    phase = 2 * np.pi * np.cumsum(freq) / sample_rate
    
    signal = 0.75 * np.sin(phase) + 0.2 * np.sin(3 * phase)
    signal = (signal / np.max(np.abs(signal))).astype(np.float32)

    if start_sec is not None or end_sec is not None or duration_sec > 5.0:
        signal = apply_burst_envelope(signal, sample_rate, duration_sec, start_sec, end_sec)

    return signal


def generate_traffic_noise(duration_sec=10.0, sample_rate=16000, noise_type="rumble"):
    """
    Synthesize continuous background traffic noise for given duration.
    """
    num_samples = int(sample_rate * duration_sec)
    t = np.linspace(0, duration_sec, num_samples, endpoint=False)
    
    unequal_noise = np.random.randn(num_samples)
    fft_noise = np.fft.rfft(unequal_noise)
    frequencies = np.fft.rfftfreq(num_samples, 1 / sample_rate)
    frequencies[0] = 1.0
    pink_filter = 1 / np.sqrt(frequencies)
    pink_filter /= np.max(pink_filter)
    
    pink_fft = fft_noise * pink_filter
    pink_noise = np.fft.irfft(pink_fft, n=num_samples)
    
    engine_mod = 0.5 + 0.5 * np.sin(2 * np.pi * 0.2 * t + np.random.rand() * 2 * np.pi)
    traffic_sound = pink_noise * engine_mod
    
    if noise_type == "horns":
        # Add car horn bursts at random intervals
        for h_start_ratio in [0.2, 0.6]:
            horn_start = int(h_start_ratio * num_samples)
            horn_len = int(0.5 * sample_rate)
            if horn_start + horn_len < num_samples:
                t_horn = t[horn_start:horn_start + horn_len]
                horn = 0.35 * (np.sin(2 * np.pi * 440 * t_horn) + np.sin(2 * np.pi * 554 * t_horn))
                traffic_sound[horn_start:horn_start + horn_len] += horn

    traffic_sound = traffic_sound / (np.max(np.abs(traffic_sound)) + 1e-6)
    return traffic_sound.astype(np.float32)


def generate_sample_dataset(num_samples=10, sample_rate=16000, duration_sec=10.0):
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
        start_s = round(1.0 + (i % 3) * 1.5, 1)
        end_s = round(start_s + 4.0, 1)
        s_audio = s_gen(duration_sec=duration_sec, sample_rate=sample_rate, start_sec=start_s, end_sec=end_s)
        file_path = SIREN_DIR / f"siren_{s_name}_{i+1:02d}.wav"
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
