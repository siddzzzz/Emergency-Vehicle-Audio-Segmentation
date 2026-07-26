import os
import random
import torch
from torch.utils.data import Dataset
import numpy as np
import scipy.io.wavfile as wavfile
from pathlib import Path

from src.data_generator import generate_siren_wail, generate_siren_yelp, generate_siren_hilo, generate_traffic_noise

BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR = BASE_DIR / "data"


class AudioProcessor:
    """
    STFT and iSTFT helper for audio spectrogram transform and waveform reconstruction.
    Uses PyTorch native torch.stft and torch.istft.
    """
    def __init__(self, sample_rate=16000, n_fft=512, hop_length=160, win_length=400):
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.window = torch.hann_window(win_length)

    def stft(self, waveform):
        """
        Computes STFT of 1D waveform tensor.
        Returns:
            mag: Spectrogram magnitude tensor of shape (1, Freq, Time)
            phase: Spectrogram phase angle tensor of shape (1, Freq, Time)
        """
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
            
        stft_complex = torch.stft(
            waveform,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window.to(waveform.device),
            return_complex=True
        )
        mag = torch.abs(stft_complex)
        phase = torch.angle(stft_complex)
        return mag, phase

    def istft(self, mag, phase):
        """
        Reconstructs 1D time-domain waveform from Magnitude and Phase tensors.
        """
        stft_complex = mag * torch.exp(1j * phase)
        if stft_complex.dim() == 3 and stft_complex.size(0) == 1:
            stft_complex = stft_complex.squeeze(0)
            
        waveform = torch.istft(
            stft_complex,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window.to(mag.device)
        )
        return waveform


class EmergencyAudioDataset(Dataset):
    """
    Dynamic Audio Dataset:
    Overlays emergency vehicle siren onto traffic noise at randomized SNR (-10dB to +10dB).
    Relative paths are used by default.
    """
    def __init__(
        self,
        data_dir=None,
        sample_rate=16000,
        duration_sec=4.0,
        snr_range=(-10.0, 10.0),
        dataset_size=200,
        n_fft=512,
        hop_length=160,
        win_length=400
    ):
        super().__init__()
        self.data_dir = Path(data_dir) if data_dir else DEFAULT_DATA_DIR
        self.sample_rate = sample_rate
        self.num_samples = int(sample_rate * duration_sec)
        self.snr_range = snr_range
        self.dataset_size = dataset_size
        self.processor = AudioProcessor(sample_rate, n_fft, hop_length, win_length)

        self.siren_files = list((self.data_dir / "sirens").glob("*.wav")) if (self.data_dir / "sirens").exists() else []
        self.traffic_files = list((self.data_dir / "traffic").glob("*.wav")) if (self.data_dir / "traffic").exists() else []

    def _load_or_generate_audio(self, file_list, gen_type="siren"):
        if file_list:
            filepath = random.choice(file_list)
            sr, audio_np = wavfile.read(str(filepath))
            # Convert to float32 [-1.0, 1.0]
            if audio_np.dtype == np.int16:
                audio_np = audio_np.astype(np.float32) / 32768.0
            elif audio_np.dtype == np.int32:
                audio_np = audio_np.astype(np.float32) / 2147483648.0
            else:
                audio_np = audio_np.astype(np.float32)
                
            if audio_np.ndim > 1:
                audio_np = audio_np.mean(axis=1)  # Mono
                
            waveform = torch.tensor(audio_np, dtype=torch.float32)
            
            if waveform.size(0) < self.num_samples:
                pad_len = self.num_samples - waveform.size(0)
                waveform = torch.nn.functional.pad(waveform, (0, pad_len))
            elif waveform.size(0) > self.num_samples:
                start = random.randint(0, waveform.size(0) - self.num_samples)
                waveform = waveform[start:start + self.num_samples]
            return waveform
        else:
            if gen_type == "siren":
                gen_fn = random.choice([generate_siren_wail, generate_siren_yelp, generate_siren_hilo])
                audio = gen_fn(duration_sec=4.0, sample_rate=self.sample_rate)
            else:
                n_type = "horns" if random.random() > 0.5 else "rumble"
                audio = generate_traffic_noise(duration_sec=4.0, sample_rate=self.sample_rate, noise_type=n_type)
            return torch.tensor(audio, dtype=torch.float32)

    def mix_audio(self, siren, traffic, snr_db):
        """
        Mixes siren and traffic noise based on target Signal-to-Noise Ratio (SNR) in dB.
        """
        p_siren = torch.mean(siren ** 2) + 1e-8
        p_traffic = torch.mean(traffic ** 2) + 1e-8
        
        target_siren_power = p_traffic * (10.0 ** (snr_db / 10.0))
        scale = torch.sqrt(target_siren_power / p_siren)
        scaled_siren = siren * scale
        
        mixture = traffic + scaled_siren
        # Normalize to avoid clipping
        max_val = torch.max(torch.abs(mixture))
        if max_val > 1.0:
            mixture = mixture / max_val
            scaled_siren = scaled_siren / max_val
            
        return mixture, scaled_siren

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        siren_raw = self._load_or_generate_audio(self.siren_files, gen_type="siren")
        traffic_raw = self._load_or_generate_audio(self.traffic_files, gen_type="traffic")
        
        snr_db = random.uniform(self.snr_range[0], self.snr_range[1])
        mixture_wave, clean_siren_wave = self.mix_audio(siren_raw, traffic_raw, snr_db)

        # STFT Spectrogram calculation
        mix_mag, mix_phase = self.processor.stft(mixture_wave)
        siren_mag, siren_phase = self.processor.stft(clean_siren_wave)

        return {
            "mix_mag": mix_mag,             # Tensor (1, Freq, Time)
            "mix_phase": mix_phase,         # Tensor (1, Freq, Time)
            "siren_mag": siren_mag,         # Tensor (1, Freq, Time)
            "mix_wave": mixture_wave,       # Waveform (Time,)
            "siren_wave": clean_siren_wave,  # Waveform (Time,)
            "snr_db": torch.tensor(snr_db, dtype=torch.float32)
        }


if __name__ == "__main__":
    dataset = EmergencyAudioDataset(dataset_size=5)
    sample = dataset[0]
    print("Dataset Test:")
    print("mix_mag shape:", sample["mix_mag"].shape)
    print("siren_mag shape:", sample["siren_mag"].shape)
    print("mix_wave shape:", sample["mix_wave"].shape)
    print("SNR (dB):", sample["snr_db"].item())
