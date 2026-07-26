import os
import sys
from pathlib import Path
import numpy as np
import scipy.io.wavfile as wavfile
import matplotlib.pyplot as plt

import torch

from src.data_generator import (
    generate_siren_wail,
    generate_siren_yelp,
    generate_siren_hilo,
    generate_traffic_noise
)
from src.dataset import AudioProcessor
from src.model import SpectrogramUNet

# Relative directory paths
BASE_DIR = Path(__file__).resolve().parent
CKPT_PATH = BASE_DIR / "checkpoints" / "best_model.pth"
TEMP_DIR = BASE_DIR / "temp_audio"
TEMP_DIR.mkdir(exist_ok=True)

# Initialize Processor and Model
SAMPLE_RATE = 16000
processor = AudioProcessor(sample_rate=SAMPLE_RATE)
device = "cuda" if torch.cuda.is_available() else "cpu"

model = SpectrogramUNet().to(device)
model.eval()

if CKPT_PATH.exists():
    try:
        ckpt = torch.load(CKPT_PATH, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"Loaded trained model checkpoint from {CKPT_PATH}")
    except Exception as e:
        print(f"Notice: Could not load checkpoint ({e}). Using initialized model.")
else:
    print("Notice: No trained checkpoint found yet. Interface running with default model.")


def plot_spectrograms(mix_mag, siren_pred_mag, clean_siren_mag=None):
    """
    Generates Matplotlib figure comparing STFT magnitude spectrograms.
    """
    fig, axes = plt.subplots(1, 3 if clean_siren_mag is not None else 2, figsize=(14, 4))
    
    # Log scale for spectrogram visualization
    eps = 1e-6
    mix_db = 20 * np.log10(mix_mag + eps)
    pred_db = 20 * np.log10(siren_pred_mag + eps)

    im0 = axes[0].imshow(mix_db, aspect="auto", origin="lower", cmap="magma")
    axes[0].set_title("Mixed Traffic + Siren Spectrogram")
    axes[0].set_xlabel("Time Frames")
    axes[0].set_ylabel("Frequency Bins")
    fig.colorbar(im0, ax=axes[0], format="%+2.0f dB")

    im1 = axes[1].imshow(pred_db, aspect="auto", origin="lower", cmap="magma")
    axes[1].set_title("Extracted Emergency Siren (Model)")
    axes[1].set_xlabel("Time Frames")
    axes[1].set_ylabel("Frequency Bins")
    fig.colorbar(im1, ax=axes[1], format="%+2.0f dB")

    if clean_siren_mag is not None:
        gt_db = 20 * np.log10(clean_siren_mag + eps)
        im2 = axes[2].imshow(gt_db, aspect="auto", origin="lower", cmap="magma")
        axes[2].set_title("Ground Truth Siren Spectrogram")
        axes[2].set_xlabel("Time Frames")
        axes[2].set_ylabel("Frequency Bins")
        fig.colorbar(im2, ax=axes[2], format="%+2.0f dB")

    plt.tight_layout()
    return fig


def process_audio_separation(siren_type, noise_type, snr_db, custom_audio_file=None):
    """
    Gradio Event Handler: Mixes siren & traffic noise, runs PyTorch model separation,
    and returns audio outputs + spectrogram plots.
    """
    duration = 4.0
    num_samples = int(SAMPLE_RATE * duration)

    if custom_audio_file is not None:
        # Load user custom input audio
        sr, audio_np = wavfile.read(custom_audio_file)
        if audio_np.dtype == np.int16:
            audio_np = audio_np.astype(np.float32) / 32768.0
        elif audio_np.dtype == np.int32:
            audio_np = audio_np.astype(np.float32) / 2147483648.0
        else:
            audio_np = audio_np.astype(np.float32)

        if audio_np.ndim > 1:
            audio_np = audio_np.mean(axis=1)

        mix_wave = torch.tensor(audio_np[:num_samples], dtype=torch.float32)
        clean_siren_wave = None
        traffic_wave = None
    else:
        # Synthesize selected Siren & Traffic
        if siren_type == "Wail (Ambulance)":
            siren_np = generate_siren_wail(duration_sec=duration, sample_rate=SAMPLE_RATE)
        elif siren_type == "Yelp (Police)":
            siren_np = generate_siren_yelp(duration_sec=duration, sample_rate=SAMPLE_RATE)
        else:
            siren_np = generate_siren_hilo(duration_sec=duration, sample_rate=SAMPLE_RATE)

        traffic_np = generate_traffic_noise(
            duration_sec=duration,
            sample_rate=SAMPLE_RATE,
            noise_type="horns" if noise_type == "Traffic + Horns" else "rumble"
        )

        siren_tensor = torch.tensor(siren_np, dtype=torch.float32)
        traffic_tensor = torch.tensor(traffic_np, dtype=torch.float32)

        # Apply SNR mixing
        p_siren = torch.mean(siren_tensor ** 2) + 1e-8
        p_traffic = torch.mean(traffic_tensor ** 2) + 1e-8
        target_siren_power = p_traffic * (10.0 ** (snr_db / 10.0))
        scale = torch.sqrt(target_siren_power / p_siren)

        clean_siren_wave = siren_tensor * scale
        traffic_wave = traffic_tensor
        mix_wave = traffic_wave + clean_siren_wave

        # Normalize to prevent audio clipping
        max_val = torch.max(torch.abs(mix_wave))
        if max_val > 1.0:
            mix_wave = mix_wave / max_val
            clean_siren_wave = clean_siren_wave / max_val
            traffic_wave = traffic_wave / max_val

    # Model Spectrogram STFT Inference
    mix_mag, mix_phase = processor.stft(mix_wave)
    mix_mag_input = mix_mag.unsqueeze(0).to(device)

    with torch.no_grad():
        mask, pred_mag = model(mix_mag_input)

    pred_mag = pred_mag.squeeze(0).cpu()
    mix_mag_cpu = mix_mag.squeeze(0).cpu()

    # Reconstruct extracted siren time-domain waveform
    est_siren_wave = processor.istft(pred_mag, mix_phase)
    est_siren_np = est_siren_wave.cpu().numpy()
    mix_np = mix_wave.cpu().numpy()

    # Save temp wav files for Gradio Audio Components
    mix_file = str(TEMP_DIR / "mixed_input.wav")
    est_file = str(TEMP_DIR / "extracted_siren.wav")

    wavfile.write(mix_file, SAMPLE_RATE, (mix_np * 32767).astype(np.int16))
    wavfile.write(est_file, SAMPLE_RATE, (np.clip(est_siren_np, -1.0, 1.0) * 32767).astype(np.int16))

    clean_siren_mag_np = None
    if clean_siren_wave is not None:
        gt_siren_file = str(TEMP_DIR / "ground_truth_siren.wav")
        wavfile.write(gt_siren_file, SAMPLE_RATE, (clean_siren_wave.numpy() * 32767).astype(np.int16))
        clean_mag, _ = processor.stft(clean_siren_wave)
        clean_siren_mag_np = clean_mag.squeeze(0).cpu().numpy()

    fig = plot_spectrograms(
        mix_mag_cpu.numpy()[0],
        pred_mag.numpy()[0],
        clean_siren_mag_np[0] if clean_siren_mag_np is not None else None
    )

    return mix_file, est_file, fig


def launch_app():
    try:
        import gradio as gr
    except ImportError:
        print("Gradio is not installed. To run the web interface, install gradio with:")
        print("   pip install gradio")
        return

    theme = gr.themes.Soft(primary_hue="red", secondary_hue="slate")
    with gr.Blocks(theme=theme, title="Emergency Vehicle Audio Separation") as demo:
        gr.Markdown(
            """
            # 🚨 Emergency Vehicle Audio Segmentation & Separation
            Isolate emergency vehicle sirens (Ambulance, Police, Firetruck) from loud traffic noise using a 2D Spectrogram Ratio-Masking U-Net.
            """
        )

        with gr.Tab("🎧 Interactive Audio Separator"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### 1. Synthetic Audio Mixture Settings")
                    siren_type = gr.Dropdown(
                        choices=["Wail (Ambulance)", "Yelp (Police)", "Hi-Lo (Firetruck)"],
                        value="Wail (Ambulance)",
                        label="Emergency Siren Sound"
                    )
                    noise_type = gr.Dropdown(
                        choices=["Heavy Traffic Rumble", "Traffic + Horns"],
                        value="Traffic + Horns",
                        label="Background Traffic Noise"
                    )
                    snr_slider = gr.Slider(
                        minimum=-12.0,
                        maximum=12.0,
                        value=0.0,
                        step=1.0,
                        label="Signal-to-Noise Ratio (SNR in dB)",
                        info="Lower dB = Traffic is much louder than the siren"
                    )
                    custom_file = gr.File(
                        label="Or Upload Custom Traffic+Siren Audio (.wav)",
                        file_types=[".wav"]
                    )
                    separate_btn = gr.Button("⚡ Separate Emergency Siren", variant="primary")

                with gr.Column(scale=2):
                    gr.Markdown("### 2. Audio & Spectrogram Outputs")
                    with gr.Row():
                        audio_mix = gr.Audio(label="Input Mixed Audio (Traffic + Siren)", type="filepath")
                        audio_separated = gr.Audio(label="Extracted Emergency Siren (Model Output)", type="filepath")

                    spec_plot = gr.Plot(label="STFT Spectrogram Comparison")

            separate_btn.click(
                fn=process_audio_separation,
                inputs=[siren_type, noise_type, snr_slider, custom_file],
                outputs=[audio_mix, audio_separated, spec_plot]
            )

        with gr.Tab("📘 Model Architecture & Learning Info"):
            gr.Markdown(
                """
                ### Audio Source Separation Mechanics:
                1. **Short-Time Fourier Transform (STFT)**: Converts 1D raw waveform into 2D time-frequency Magnitude $|X_{\\text{mix}}|$ & Phase $\\theta$.
                2. **Spectrogram Ratio Masking**: The Spectrogram U-Net outputs a soft ratio mask $M(f, t) \\in [0, 1]$.
                3. **Signal Reconstruction**: The estimated emergency siren spectrum is $\\hat{|S|} = M \\odot |X_{\\text{mix}}|$. Applying `iSTFT` with the original phase yields the isolated siren sound.
                4. **SI-SDR Metric**: Scale-Invariant Signal-to-Distortion Ratio evaluates signal purity independently of gain.
                """
            )

    print("Launching Gradio App...")
    demo.launch(share=False)


if __name__ == "__main__":
    launch_app()
