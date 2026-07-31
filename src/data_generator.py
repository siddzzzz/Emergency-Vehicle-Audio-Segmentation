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


def apply_doppler_effect(signal, sample_rate=16000, duration_sec=10.0, vehicle_speed_kmh=60.0, closest_approach_sec=5.0, min_distance_m=10.0):
    """
    Applies physical Doppler pitch shift and inverse distance volume gain to siren audio as vehicle moves.
    
    Args:
        signal: 1D NumPy array of siren audio
        sample_rate: 16000 Hz
        duration_sec: Total duration in seconds (10.0s)
        vehicle_speed_kmh: Vehicle speed in km/h (e.g. 60 km/h = 16.67 m/s)
        closest_approach_sec: Time in seconds when vehicle is closest to junction (5.0s)
        min_distance_m: Closest approach distance to microphone array in meters (10.0m)
    """
    if vehicle_speed_kmh <= 0.0:
        return signal

    v_sound = 343.0  # Speed of sound in m/s
    v_speed = vehicle_speed_kmh * (1000.0 / 3600.0)  # Convert km/h to m/s

    num_samples = len(signal)
    t = np.linspace(0, duration_sec, num_samples, endpoint=False)

    # Position x(t) relative to closest approach point (x = 0 at closest_approach_sec)
    x = v_speed * (t - closest_approach_sec)
    
    # Distance r(t) from vehicle to microphone array at (0, min_distance_m)
    r = np.sqrt(x**2 + min_distance_m**2)

    # Distance gain (Inverse Distance Law 1/r) normalized to 1.0 at min_distance_m
    distance_gain = min_distance_m / r

    # Radial velocity component toward microphone array
    # v_radial is positive when approaching (moving toward mic), negative when receding
    v_radial = -v_speed * (x / r)

    # Doppler frequency scaling factor f_observed / f_source = v_sound / (v_sound - v_radial)
    doppler_factor = v_sound / (v_sound - v_radial + 1e-6)

    # Dynamic time warp: dt_observed = dt_source * doppler_factor
    # Integrated sample mapping to resample signal continuously
    integrated_time = np.cumsum(doppler_factor) / sample_rate
    # Normalize integrated time to span original sample range
    source_indices = np.interp(
        t,
        np.linspace(0, duration_sec, num_samples),
        integrated_time * (num_samples / (integrated_time[-1] + 1e-8))
    )
    source_indices = np.clip(source_indices, 0, num_samples - 1)

    # Interpolate signal values at warped time indices
    doppler_signal = np.interp(source_indices, np.arange(num_samples), signal)

    # Apply inverse distance volume gain
    doppler_signal = doppler_signal * distance_gain

    return doppler_signal.astype(np.float32)


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
    fade_len = int(0.3 * sample_rate)

    envelope = np.zeros(num_samples, dtype=np.float32)
    active_len = max(0, end_idx - start_idx)

    if active_len > 2 * fade_len:
        fade_in = 0.5 * (1.0 - np.cos(np.pi * np.linspace(0, 1, fade_len)))
        fade_out = 0.5 * (1.0 + np.cos(np.pi * np.linspace(0, 1, fade_len)))
        
        envelope[start_idx:start_idx + fade_len] = fade_in
        envelope[start_idx + fade_len:end_idx - fade_len] = 1.0
        envelope[end_idx - fade_len:end_idx] = fade_out
    else:
        envelope[start_idx:end_idx] = 1.0

    return (signal * envelope).astype(np.float32)


def generate_siren_wail(duration_sec=10.0, sample_rate=16000, start_sec=None, end_sec=None, vehicle_speed_kmh=0.0, closest_approach_sec=5.0):
    """
    Synthesize an Ambulance/Police Wail Siren with optional Doppler pitch shift and burst envelope.
    """
    num_samples = int(sample_rate * duration_sec)
    t = np.linspace(0, duration_sec, num_samples, endpoint=False)
    mod_freq = 0.3
    f_min, f_max = 600.0, 1300.0
    
    freq_inst = f_min + (f_max - f_min) * 0.5 * (1 + np.sin(2 * np.pi * mod_freq * t - np.pi / 2))
    phase = 2 * np.pi * np.cumsum(freq_inst) / sample_rate
    
    signal = 0.7 * np.sin(phase) + 0.2 * np.sin(2 * phase)
    signal = (signal / np.max(np.abs(signal))).astype(np.float32)

    if vehicle_speed_kmh > 0.0:
        signal = apply_doppler_effect(signal, sample_rate, duration_sec, vehicle_speed_kmh, closest_approach_sec)

    if start_sec is not None or end_sec is not None or duration_sec > 5.0:
        signal = apply_burst_envelope(signal, sample_rate, duration_sec, start_sec, end_sec)

    return signal


def generate_siren_yelp(duration_sec=10.0, sample_rate=16000, start_sec=None, end_sec=None, vehicle_speed_kmh=0.0, closest_approach_sec=5.0):
    """
    Synthesize a Yelp Siren with optional Doppler pitch shift and burst envelope.
    """
    num_samples = int(sample_rate * duration_sec)
    t = np.linspace(0, duration_sec, num_samples, endpoint=False)
    mod_freq = 2.5
    f_min, f_max = 650.0, 1450.0
    
    freq_inst = f_min + (f_max - f_min) * 0.5 * (1 + np.sin(2 * np.pi * mod_freq * t))
    phase = 2 * np.pi * np.cumsum(freq_inst) / sample_rate
    
    signal = 0.8 * np.sin(phase) + 0.15 * np.sin(2 * phase)
    signal = (signal / np.max(np.abs(signal))).astype(np.float32)

    if vehicle_speed_kmh > 0.0:
        signal = apply_doppler_effect(signal, sample_rate, duration_sec, vehicle_speed_kmh, closest_approach_sec)

    if start_sec is not None or end_sec is not None or duration_sec > 5.0:
        signal = apply_burst_envelope(signal, sample_rate, duration_sec, start_sec, end_sec)

    return signal


def generate_siren_hilo(duration_sec=10.0, sample_rate=16000, start_sec=None, end_sec=None, vehicle_speed_kmh=0.0, closest_approach_sec=5.0):
    """
    Synthesize a European / Firetruck Hi-Lo Siren with optional Doppler pitch shift and burst envelope.
    """
    num_samples = int(sample_rate * duration_sec)
    t = np.linspace(0, duration_sec, num_samples, endpoint=False)
    period = 1.0
    freq = np.where((t % period) < 0.5, 900.0, 700.0)
    phase = 2 * np.pi * np.cumsum(freq) / sample_rate
    
    signal = 0.75 * np.sin(phase) + 0.2 * np.sin(3 * phase)
    signal = (signal / np.max(np.abs(signal))).astype(np.float32)

    if vehicle_speed_kmh > 0.0:
        signal = apply_doppler_effect(signal, sample_rate, duration_sec, vehicle_speed_kmh, closest_approach_sec)

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
        speed = round(40.0 + (i % 4) * 15.0, 1)
        s_audio = s_gen(duration_sec=duration_sec, sample_rate=sample_rate, start_sec=start_s, end_sec=end_s, vehicle_speed_kmh=speed)
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
