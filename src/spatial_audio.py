import numpy as np
import scipy.signal as signal
from pathlib import Path


class SpatialAudioEngine:
    """
    Spatial Audio & Direction of Arrival (DoA) Estimation Engine for Traffic Intersections.
    Simulates multi-microphone acoustic arrays at a 4-way traffic stop and estimates
    the approaching lane direction of emergency vehicles using GCC-PHAT cross-correlation.
    """
    def __init__(self, mic_distance=0.5, sample_rate=16000, speed_of_sound=343.0):
        self.mic_distance = mic_distance        # Distance between Mic 1 and Mic 2 in meters (e.g. 0.5m)
        self.sample_rate = sample_rate          # 16000 Hz
        self.speed_of_sound = speed_of_sound    # 343 m/s in air
        self.max_delay_sec = self.mic_distance / self.speed_of_sound
        self.max_delay_samples = int(np.ceil(self.max_delay_sec * self.sample_rate))

    def apply_time_delay(self, waveform, delay_sec):
        """
        Applies a sub-sample time delay to 1D waveform using sinc interpolation / FFT phase shift.
        """
        num_samples = len(waveform)
        fft_data = np.fft.rfft(waveform)
        freqs = np.fft.rfftfreq(num_samples, 1.0 / self.sample_rate)
        
        # Linear phase shift corresponding to time delay
        phase_shift = np.exp(-1j * 2.0 * np.pi * freqs * delay_sec)
        delayed_fft = fft_data * phase_shift
        delayed_wave = np.fft.irfft(delayed_fft, n=num_samples)
        return delayed_wave.astype(np.float32)

    def simulate_stereo_mixture(self, siren_wave, traffic_wave, angle_deg=0.0):
        """
        Simulates 2-channel stereo audio (Mic 1 & Mic 2) at a traffic stop.
        
        Args:
            siren_wave: 1D NumPy array of siren audio
            traffic_wave: 1D NumPy array of background traffic noise
            angle_deg: Angle of approaching emergency vehicle in degrees (-90 to +90)
                       0 deg = Lane 1 (Center / North)
                       +45 deg = Lane 2 (Right / East)
                       -45 deg = Lane 3 (Left / West)
        Returns:
            mic1_mix: 1D array of mixed audio at Mic 1
            mic2_mix: 1D array of mixed audio at Mic 2
        """
        angle_rad = np.radians(angle_deg)
        # Time difference of arrival (TDoA) between Mic 1 and Mic 2
        tdoa = (self.mic_distance * np.sin(angle_rad)) / self.speed_of_sound

        # Mic 1 is reference (0 delay), Mic 2 receives siren with tdoa delay
        siren_mic1 = siren_wave
        siren_mic2 = self.apply_time_delay(siren_wave, tdoa)

        # Diffuse background traffic noise (independent noise at Mic 1 and Mic 2)
        traffic_mic1 = traffic_wave
        # Independent phase shift for traffic background
        traffic_mic2 = self.apply_time_delay(traffic_wave, 0.0005)

        mic1_mix = traffic_mic1 + siren_mic1
        mic2_mix = traffic_mic2 + siren_mic2

        return mic1_mix, mic2_mix, siren_mic1, siren_mic2

    def estimate_doa_gcc_phat(self, mic1_wave, mic2_wave):
        """
        Generalized Cross-Correlation with Phase Transform (GCC-PHAT)
        Estimates the exact time delay (TDoA) and angle theta between Mic 1 and Mic 2.
        
        Returns:
            angle_deg: Estimated angle of arrival in degrees (-90 to +90)
            lane_name: Target traffic lane decision string
            confidence_score: Correlation peak sharpness
        """
        n = len(mic1_wave) + len(mic2_wave)

        # FFT of mic 1 and mic 2 signals
        X1 = np.fft.rfft(mic1_wave, n=n)
        X2 = np.fft.rfft(mic2_wave, n=n)

        # Cross power spectrum with phase transform (GCC-PHAT)
        R = X1 * np.conj(X2)
        R_phat = R / (np.abs(R) + 1e-6)

        # Cross-correlation function in time domain
        cc = np.fft.irfft(R_phat, n=n)

        # Search range centered at lag 0
        cc_shifted = np.concatenate((cc[-self.max_delay_samples:], cc[:self.max_delay_samples + 1]))
        delay_indices = np.arange(-self.max_delay_samples, self.max_delay_samples + 1)

        max_idx = np.argmax(cc_shifted)
        peak_val = float(cc_shifted[max_idx])
        
        # Parabolic interpolation around peak for sub-sample accuracy
        if 0 < max_idx < len(cc_shifted) - 1:
            y0 = cc_shifted[max_idx - 1]
            y1 = cc_shifted[max_idx]
            y2 = cc_shifted[max_idx + 1]
            denom = (2.0 * (2.0 * y1 - y0 - y2)) + 1e-8
            delta = (y0 - y2) / denom
            delay_samples = delay_indices[max_idx] + delta
        else:
            delay_samples = float(delay_indices[max_idx])

        # Convert delay in samples to time delay in seconds
        estimated_tdoa = delay_samples / float(self.sample_rate)

        # Compute angle theta = arcsin(tdoa * v / d)
        sin_theta = (estimated_tdoa * self.speed_of_sound) / self.mic_distance
        sin_theta = np.clip(sin_theta, -1.0, 1.0)
        angle_deg = float(np.degrees(np.arcsin(sin_theta)))

        # Classify exact traffic lane based on angle
        if angle_deg > 18.0:
            lane_name = "Lane 2 (Eastbound Right)"
        elif angle_deg < -18.0:
            lane_name = "Lane 3 (Westbound Left)"
        else:
            lane_name = "Lane 1 (Northbound Center)"

        return round(angle_deg, 1), lane_name, peak_val
