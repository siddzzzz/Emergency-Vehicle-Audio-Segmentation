import os
import sys
from pathlib import Path
import numpy as np
import scipy.io.wavfile as wavfile
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for server rendering
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
        print(f"Notice: Could not load checkpoint ({e}). Running with default initialized model.")
else:
    print("Notice: No trained checkpoint found yet. Running with default initialized model.")


def plot_spectrograms_and_mask(mix_mag, mask, siren_pred_mag, clean_siren_mag=None):
    """
    Generates Matplotlib figure comparing:
    1. Mixed Traffic + Siren Spectrogram
    2. Predicted U-Net Ratio Mask M in [0, 1]
    3. Extracted Emergency Siren Spectrogram
    4. Ground Truth Siren Spectrogram (if available)
    """
    num_plots = 4 if clean_siren_mag is not None else 3
    fig, axes = plt.subplots(1, num_plots, figsize=(16, 4.2), dpi=120)

    eps = 1e-6
    mix_db = 20 * np.log10(mix_mag + eps)
    pred_db = 20 * np.log10(siren_pred_mag + eps)

    # 1. Mixed Input Spectrogram
    im0 = axes[0].imshow(mix_db, aspect="auto", origin="lower", cmap="inferno")
    axes[0].set_title("1. Mixed Input (Traffic + Siren)", fontsize=11, fontweight="bold")
    axes[0].set_xlabel("Time Frames")
    axes[0].set_ylabel("Frequency Bins")
    fig.colorbar(im0, ax=axes[0], format="%+2.0f dB")

    # 2. Predicted Ratio Mask M(f, t)
    im1 = axes[1].imshow(mask, aspect="auto", origin="lower", cmap="viridis", vmin=0.0, vmax=1.0)
    axes[1].set_title("2. Predicted Ratio Mask M(f,t)", fontsize=11, fontweight="bold")
    axes[1].set_xlabel("Time Frames")
    axes[1].set_ylabel("Frequency Bins")
    cbar1 = fig.colorbar(im1, ax=axes[1])
    cbar1.set_label("Mask Value (0=Noise, 1=Siren)", fontsize=9)

    # 3. Extracted Siren Spectrogram
    im2 = axes[2].imshow(pred_db, aspect="auto", origin="lower", cmap="inferno")
    axes[2].set_title("3. Extracted Siren (Model Output)", fontsize=11, fontweight="bold")
    axes[2].set_xlabel("Time Frames")
    axes[2].set_ylabel("Frequency Bins")
    fig.colorbar(im2, ax=axes[2], format="%+2.0f dB")

    # 4. Ground Truth Siren Spectrogram (if synthetic)
    if clean_siren_mag is not None:
        gt_db = 20 * np.log10(clean_siren_mag + eps)
        im3 = axes[3].imshow(gt_db, aspect="auto", origin="lower", cmap="inferno")
        axes[3].set_title("4. Ground Truth Siren", fontsize=11, fontweight="bold")
        axes[3].set_xlabel("Time Frames")
        axes[3].set_ylabel("Frequency Bins")
        fig.colorbar(im3, ax=axes[3], format="%+2.0f dB")

    plt.tight_layout()
    return fig


def safe_extract_filepath(file_obj):
    """Safely extracts filepath string from Gradio file input across different versions."""
    if file_obj is None:
        return None
    if isinstance(file_obj, str):
        return file_obj
    if isinstance(file_obj, dict) and "name" in file_obj:
        return file_obj["name"]
    if hasattr(file_obj, "name"):
        return file_obj.name
    return str(file_obj)


def process_audio_separation(siren_type, noise_type, snr_db, custom_audio_file=None):
    """
    Gradio Event Handler: Performs dynamic audio mixing, STFT processing,
    ratio-mask inference, and outputs 4 audio tracks + spectrogram comparison plot.
    """
    duration = 4.0
    num_samples = int(SAMPLE_RATE * duration)
    custom_path = safe_extract_filepath(custom_audio_file)

    clean_siren_wave = None
    traffic_wave = None

    if custom_path and os.path.exists(custom_path):
        # Load user custom WAV file
        sr, audio_np = wavfile.read(custom_path)
        if audio_np.dtype == np.int16:
            audio_np = audio_np.astype(np.float32) / 32768.0
        elif audio_np.dtype == np.int32:
            audio_np = audio_np.astype(np.float32) / 2147483648.0
        else:
            audio_np = audio_np.astype(np.float32)

        if audio_np.ndim > 1:
            audio_np = audio_np.mean(axis=1)

        # Pad or crop to exact 4 seconds
        if len(audio_np) < num_samples:
            audio_np = np.pad(audio_np, (0, num_samples - len(audio_np)))
        else:
            audio_np = audio_np[:num_samples]

        mix_wave = torch.tensor(audio_np, dtype=torch.float32)
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
    mix_mag, mix_phase = processor.stft(mix_wave)  # shape (1, 257, 401)
    mix_mag_input = mix_mag.unsqueeze(0).to(device)  # shape (1, 1, 257, 401)

    with torch.no_grad():
        mask_tensor, pred_mag_tensor = model(mix_mag_input)

    mask_np = mask_tensor.squeeze().cpu().numpy()  # shape (257, 401)
    pred_mag_np = pred_mag_tensor.squeeze().cpu().numpy()  # shape (257, 401)
    mix_mag_np = mix_mag.squeeze().cpu().numpy()  # shape (257, 401)

    # Reconstruct time-domain audio waveforms using iSTFT
    est_siren_wave = processor.istft(pred_mag_tensor.squeeze(0).cpu(), mix_phase)
    est_siren_np = est_siren_wave.cpu().numpy()
    mix_np = mix_wave.cpu().numpy()

    # Save audio files for Gradio components
    mix_file = str(TEMP_DIR / "mixed_input.wav")
    est_file = str(TEMP_DIR / "extracted_siren.wav")
    gt_siren_file = str(TEMP_DIR / "ground_truth_siren.wav")
    traffic_file = str(TEMP_DIR / "traffic_noise.wav")

    wavfile.write(mix_file, SAMPLE_RATE, (mix_np * 32767).astype(np.int16))
    wavfile.write(est_file, SAMPLE_RATE, (np.clip(est_siren_np, -1.0, 1.0) * 32767).astype(np.int16))

    clean_siren_mag_np = None
    if clean_siren_wave is not None:
        wavfile.write(gt_siren_file, SAMPLE_RATE, (clean_siren_wave.numpy() * 32767).astype(np.int16))
        wavfile.write(traffic_file, SAMPLE_RATE, (traffic_wave.numpy() * 32767).astype(np.int16))
        clean_mag, _ = processor.stft(clean_siren_wave)
        clean_siren_mag_np = clean_mag.squeeze().cpu().numpy()  # shape (257, 401)
    else:
        gt_siren_file = None
        traffic_file = None

    fig = plot_spectrograms_and_mask(
        mix_mag_np,
        mask_np,
        pred_mag_np,
        clean_siren_mag_np
    )

    status_msg = f"**Audio Separation Complete!** (SNR: `{snr_db:+.1f} dB` | Sample Rate: `{SAMPLE_RATE} Hz` | Device: `{device.upper()}`)"

    return mix_file, est_file, gt_siren_file, traffic_file, fig, status_msg


def launch_app():
    try:
        import gradio as gr
    except ImportError:
        print("Gradio is not installed. Please install it using: pip install gradio")
        return

    custom_css = """
    .main-container { max-width: 1280px; margin: 0 auto; }
    .card-box { border: 1px solid #374151; border-radius: 8px; padding: 16px; background-color: #1f2937; }
    .btn-primary { background: linear-gradient(135deg, #ef4444 0%, #b91c1c 100%) !important; color: white !important; font-weight: bold !important; }
    """

    theme = gr.themes.Soft(primary_hue="red", secondary_hue="slate")
    with gr.Blocks(theme=theme, css=custom_css, title="Emergency Vehicle Audio Separation") as demo:
        gr.Markdown(
            """
            # 🚨 Emergency Vehicle Audio Segmentation & Source Separation
            *Isolate emergency sirens (Ambulance, Police, Firetruck) from heavy traffic noise using a 2D Spectrogram U-Net.*
            """
        )

        with gr.Tab("🎧 Interactive Audio Separator"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### 🎛️ 1. Audio Mixture Controls")
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
                        info="Lower dB = Traffic is louder than siren"
                    )
                    custom_file = gr.File(
                        label="Or Upload Custom Traffic+Siren Audio (.wav)",
                        file_types=[".wav"]
                    )
                    separate_btn = gr.Button("⚡ Separate Emergency Siren", variant="primary", elem_classes=["btn-primary"])

                with gr.Column(scale=2):
                    gr.Markdown("### 📊 2. Visual Spectrogram & Ratio Mask Output")
                    status_banner = gr.Markdown("Click **Separate Emergency Siren** to run separation.")
                    spec_plot = gr.Plot(label="STFT Spectrogram & Mask Comparison")

            gr.Markdown("### 🔊 3. Audio Track Playback & Comparisons")
            with gr.Row():
                audio_mix = gr.Audio(label="1. Mixed Input (Traffic + Siren)", type="filepath")
                audio_separated = gr.Audio(label="2. Extracted Siren (Model Output)", type="filepath")
                audio_gt_siren = gr.Audio(label="3. Clean Ground-Truth Siren", type="filepath")
                audio_traffic = gr.Audio(label="4. Background Traffic Noise", type="filepath")

            separate_btn.click(
                fn=process_audio_separation,
                inputs=[siren_type, noise_type, snr_slider, custom_file],
                outputs=[audio_mix, audio_separated, audio_gt_siren, audio_traffic, spec_plot, status_banner]
            )

        with gr.Tab("📘 Architecture & Learning Guide"):
            gr.Markdown(
                """
                ### How Audio Source Separation Works:
                1. **Short-Time Fourier Transform (STFT)**: Converts 1D raw waveform $x(t)$ into a 2D Time-Frequency Spectrogram matrix $|X_{\\text{mix}}|$ & Phase $\\theta$.
                2. **2D Spectrogram U-Net**: Downsamples the spectrogram through 2D Convolutional Encoder layers, then upsamples back to predict a 2D **Ratio Mask** $M(f,t) \\in [0, 1]$.
                3. **Spectrogram Masking**: Calculates estimated siren spectrum $\\hat{|S|} = M \\odot |X_{\\text{mix}}|$.
                4. **iSTFT Waveform Reconstruction**: Applies Inverse STFT combining $\\hat{|S|}$ with phase $\\theta$ to produce the isolated time-domain emergency siren sound.
                """
            )

    print("Launching upgraded Gradio App...")
    demo.launch(share=False)


if __name__ == "__main__":
    launch_app()
