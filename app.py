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
from src.detector import SlidingWindowDetector

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

# Initialize Sliding Window Detector
detector = SlidingWindowDetector(model, processor, window_sec=4.0, hop_sec=0.5, sample_rate=SAMPLE_RATE)


def plot_detection_dashboard(mix_wave_np, time_points, confidence_curve, detected_intervals, mix_mag, mask, siren_pred_mag, clean_siren_mag=None):
    """
    Generates a 5-Subplot Visualizer Dashboard:
    1. 1D Mixed Waveform Timeline with Red Highlighted Emergency Siren Intervals
    2. Real-Time Siren Detection Confidence Curve (0% to 100%)
    3. Mixed Input Spectrogram
    4. Predicted U-Net Ratio Mask M(f, t)
    5. Extracted Siren Spectrogram
    """
    fig = plt.figure(figsize=(16, 9), dpi=120)
    gs = fig.add_gridspec(2, 4, height_ratios=[1, 1.2])

    t_axis = np.linspace(0, len(mix_wave_np) / SAMPLE_RATE, len(mix_wave_np))

    # --- Subplot 1: 1D Mixed Audio Waveform Timeline & Detection Highlights ---
    ax_wave = fig.add_subplot(gs[0, :2])
    ax_wave.plot(t_axis, mix_wave_np, color="#4b5563", alpha=0.7, label="Traffic + Siren Audio")
    
    for start_t, end_t in detected_intervals:
        ax_wave.axvspan(start_t, end_t, color="#ef4444", alpha=0.3, label="Siren Detected (Active)" if start_t == detected_intervals[0][0] else "")
    
    ax_wave.set_title("1. Mixed Traffic Audio Waveform (10s Timeline)", fontsize=11, fontweight="bold")
    ax_wave.set_xlabel("Time (seconds)")
    ax_wave.set_ylabel("Amplitude")
    ax_wave.set_xlim(0, len(mix_wave_np) / SAMPLE_RATE)
    ax_wave.grid(True, linestyle="--", alpha=0.5)
    ax_wave.legend(loc="upper right")

    # --- Subplot 2: Real-Time Siren Confidence / Detection Score Curve ---
    ax_conf = fig.add_subplot(gs[0, 2:])
    ax_conf.plot(time_points, confidence_curve, color="#dc2626", linewidth=2.5, marker="o", markersize=4, label="Siren Detection Score")
    ax_conf.axhline(detector.detection_threshold * 100.0, color="#f59e0b", linestyle="--", linewidth=1.5, label="Detection Threshold (18%)")
    
    ax_conf.set_title("2. Real-Time Siren Detection Confidence Score (%)", fontsize=11, fontweight="bold")
    ax_conf.set_xlabel("Time (seconds)")
    ax_conf.set_ylabel("Confidence (%)")
    ax_conf.set_ylim(0, 100)
    ax_conf.set_xlim(0, len(mix_wave_np) / SAMPLE_RATE)
    ax_conf.grid(True, linestyle="--", alpha=0.5)
    ax_conf.legend(loc="upper right")

    # --- Subplots 3, 4, 5: Spectrograms & Learned Ratio Mask ---
    eps = 1e-6
    mix_db = 20 * np.log10(mix_mag + eps)
    pred_db = 20 * np.log10(siren_pred_mag + eps)

    # 3. Mixed Spectrogram
    ax_spec1 = fig.add_subplot(gs[1, 0])
    im0 = ax_spec1.imshow(mix_db, aspect="auto", origin="lower", cmap="inferno")
    ax_spec1.set_title("3. Mixed Input Spectrogram", fontsize=10, fontweight="bold")
    ax_spec1.set_xlabel("Time Frames")
    ax_spec1.set_ylabel("Frequency Bins")
    fig.colorbar(im0, ax=ax_spec1, format="%+2.0f dB")

    # 4. Predicted Ratio Mask M(f,t)
    ax_mask = fig.add_subplot(gs[1, 1])
    im1 = ax_mask.imshow(mask, aspect="auto", origin="lower", cmap="viridis", vmin=0.0, vmax=1.0)
    ax_mask.set_title("4. Predicted Ratio Mask M(f,t)", fontsize=10, fontweight="bold")
    ax_mask.set_xlabel("Time Frames")
    ax_mask.set_ylabel("Frequency Bins")
    cbar1 = fig.colorbar(im1, ax=ax_mask)
    cbar1.set_label("Mask Value (0=Noise, 1=Siren)", fontsize=8)

    # 5. Extracted Siren Spectrogram
    ax_spec2 = fig.add_subplot(gs[1, 2])
    im2 = ax_spec2.imshow(pred_db, aspect="auto", origin="lower", cmap="inferno")
    ax_spec2.set_title("5. Extracted Siren Spectrogram", fontsize=10, fontweight="bold")
    ax_spec2.set_xlabel("Time Frames")
    ax_spec2.set_ylabel("Frequency Bins")
    fig.colorbar(im2, ax=ax_spec2, format="%+2.0f dB")

    # 6. Ground Truth Siren Spectrogram (if available)
    if clean_siren_mag is not None:
        ax_spec3 = fig.add_subplot(gs[1, 3])
        gt_db = 20 * np.log10(clean_siren_mag + eps)
        im3 = ax_spec3.imshow(gt_db, aspect="auto", origin="lower", cmap="inferno")
        ax_spec3.set_title("6. Ground Truth Siren Spectrogram", fontsize=10, fontweight="bold")
        ax_spec3.set_xlabel("Time Frames")
        ax_spec3.set_ylabel("Frequency Bins")
        fig.colorbar(im3, ax=ax_spec3, format="%+2.0f dB")

    plt.tight_layout()
    return fig


def safe_extract_filepath(file_obj):
    if file_obj is None:
        return None
    if isinstance(file_obj, str):
        return file_obj
    if isinstance(file_obj, dict) and "name" in file_obj:
        return file_obj["name"]
    if hasattr(file_obj, "name"):
        return file_obj.name
    return str(file_obj)


def process_audio_separation_and_detection(siren_type, noise_type, snr_db, start_sec, end_sec, custom_audio_file=None):
    """
    Gradio Event Handler:
    1. Generates 10-second continuous background traffic noise.
    2. Overlays emergency siren burst starting at `start_sec` and ending at `end_sec`.
    3. Runs Sliding Window AI Detector to compute real-time confidence scores and traffic priority override signals.
    4. Reconstructs full 10-second extracted emergency siren waveform.
    """
    duration = 10.0
    num_samples = int(SAMPLE_RATE * duration)
    custom_path = safe_extract_filepath(custom_audio_file)

    clean_siren_wave = None
    traffic_wave = None

    if custom_path and os.path.exists(custom_path):
        sr, audio_np = wavfile.read(custom_path)
        if audio_np.dtype == np.int16:
            audio_np = audio_np.astype(np.float32) / 32768.0
        elif audio_np.dtype == np.int32:
            audio_np = audio_np.astype(np.float32) / 2147483648.0
        else:
            audio_np = audio_np.astype(np.float32)

        if audio_np.ndim > 1:
            audio_np = audio_np.mean(axis=1)

        if len(audio_np) < num_samples:
            audio_np = np.pad(audio_np, (0, num_samples - len(audio_np)))
        else:
            audio_np = audio_np[:num_samples]

        mix_wave = torch.tensor(audio_np, dtype=torch.float32)
    else:
        # Generate 10-second siren burst
        if siren_type == "Wail (Ambulance)":
            siren_np = generate_siren_wail(duration_sec=duration, sample_rate=SAMPLE_RATE, start_sec=start_sec, end_sec=end_sec)
        elif siren_type == "Yelp (Police)":
            siren_np = generate_siren_yelp(duration_sec=duration, sample_rate=SAMPLE_RATE, start_sec=start_sec, end_sec=end_sec)
        else:
            siren_np = generate_siren_hilo(duration_sec=duration, sample_rate=SAMPLE_RATE, start_sec=start_sec, end_sec=end_sec)

        # Generate 10-second background traffic noise
        traffic_np = generate_traffic_noise(
            duration_sec=duration,
            sample_rate=SAMPLE_RATE,
            noise_type="horns" if noise_type == "Traffic + Horns" else "rumble"
        )

        siren_tensor = torch.tensor(siren_np, dtype=torch.float32)
        traffic_tensor = torch.tensor(traffic_np, dtype=torch.float32)

        # 1. Normalize background traffic noise to a constant, steady baseline volume (peak 0.45)
        traffic_tensor = (traffic_tensor / (torch.max(torch.abs(traffic_tensor)) + 1e-6)) * 0.45

        # 2. Calculate siren power ONLY over active non-zero frames (prevents zero-padding from distorting power)
        active_mask = torch.abs(siren_tensor) > 1e-4
        if torch.any(active_mask):
            p_siren = torch.mean(siren_tensor[active_mask] ** 2) + 1e-8
        else:
            p_siren = torch.mean(siren_tensor ** 2) + 1e-8

        p_traffic = torch.mean(traffic_tensor ** 2) + 1e-8
        target_siren_power = p_traffic * (10.0 ** (snr_db / 10.0))
        scale = torch.sqrt(target_siren_power / p_siren)

        clean_siren_wave = siren_tensor * scale
        traffic_wave = traffic_tensor  # Background traffic baseline remains 100% constant!
        mix_wave = traffic_wave + clean_siren_wave

        # Soft clip mixture if peak exceeds 1.0 to prevent hard digital clipping without ducking traffic noise
        max_mix = torch.max(torch.abs(mix_wave))
        if max_mix > 1.0:
            mix_wave = torch.tanh(mix_wave)

    # Real-Time Sliding Window Detector & Overlap-Add Separation Engine
    detection_res = detector.process_stream(mix_wave, device=device)
    
    full_est_siren_wave = detection_res["full_separated_wave"]
    time_points = detection_res["time_points"]
    confidence_curve = detection_res["confidence_curve"]
    detected_intervals = detection_res["detected_intervals"]
    traffic_action_msg = detection_res["traffic_action"]

    mix_np = mix_wave.cpu().numpy()
    est_siren_np = full_est_siren_wave.cpu().numpy()

    # Save 10-second audio files for Gradio components
    mix_file = str(TEMP_DIR / "mixed_input_10s.wav")
    est_file = str(TEMP_DIR / "extracted_siren_10s.wav")
    gt_siren_file = str(TEMP_DIR / "ground_truth_siren_10s.wav")
    traffic_file = str(TEMP_DIR / "traffic_noise_10s.wav")

    wavfile.write(mix_file, SAMPLE_RATE, (mix_np * 32767).astype(np.int16))
    wavfile.write(est_file, SAMPLE_RATE, (np.clip(est_siren_np, -1.0, 1.0) * 32767).astype(np.int16))

    clean_siren_mag_np = None
    if clean_siren_wave is not None:
        wavfile.write(gt_siren_file, SAMPLE_RATE, (clean_siren_wave.numpy() * 32767).astype(np.int16))
        wavfile.write(traffic_file, SAMPLE_RATE, (traffic_wave.numpy() * 32767).astype(np.int16))
        clean_mag, _ = processor.stft(clean_siren_wave)
        clean_siren_mag_np = clean_mag.squeeze().cpu().numpy()
    else:
        gt_siren_file = None
        traffic_file = None

    # STFT Spectrogram & Mask matrices for plotting
    mix_mag, _ = processor.stft(mix_wave)
    mix_mag_input = mix_mag.unsqueeze(0).to(device)

    with torch.no_grad():
        mask_tensor, pred_mag_tensor = model(mix_mag_input)

    mask_np = mask_tensor.squeeze().cpu().numpy()
    pred_mag_np = pred_mag_tensor.squeeze().cpu().numpy()
    mix_mag_np = mix_mag.squeeze().cpu().numpy()

    fig = plot_detection_dashboard(
        mix_np,
        time_points,
        confidence_curve,
        detected_intervals,
        mix_mag_np,
        mask_np,
        pred_mag_np,
        clean_siren_mag_np
    )

    return mix_file, est_file, gt_siren_file, traffic_file, fig, traffic_action_msg


def launch_app():
    try:
        import gradio as gr
    except ImportError:
        print("Gradio is not installed. Please install it using: pip install gradio")
        return

    custom_css = """
    .main-container { max-width: 1380px; margin: 0 auto; }
    .btn-primary { background: linear-gradient(135deg, #ef4444 0%, #b91c1c 100%) !important; color: white !important; font-weight: bold !important; font-size: 16px !important; }
    .traffic-box { border: 2px solid #ef4444; border-radius: 8px; padding: 16px; background-color: #111827; }
    """

    theme = gr.themes.Soft(primary_hue="red", secondary_hue="slate")
    with gr.Blocks(theme=theme, css=custom_css, title="Emergency Vehicle Audio Detection & Separation") as demo:
        gr.Markdown(
            """
            # 🚨 Emergency Vehicle Real-Time Audio Detection & Source Separation
            *Detect emergency vehicle sirens (Ambulance, Police, Firetruck) mid-stream in heavy traffic and isolate the siren track using a 2D Spectrogram U-Net.*
            """
        )

        with gr.Tab("🚦 Real-Time Traffic Detector & Separator"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### 🎛️ 1. Audio Stream & Siren Burst Controls")
                    siren_type = gr.Dropdown(
                        choices=["Wail (Ambulance)", "Yelp (Police)", "Hi-Lo (Firetruck)"],
                        value="Wail (Ambulance)",
                        label="Emergency Siren Sound Type"
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
                    
                    gr.Markdown("#### ⏱️ Siren Active Burst Window (Middle of 10s Stream)")
                    siren_start_slider = gr.Slider(
                        minimum=0.0,
                        maximum=6.0,
                        value=3.0,
                        step=0.5,
                        label="Siren Start Time (Seconds)",
                        info="Time when ambulance turns on its siren"
                    )
                    siren_end_slider = gr.Slider(
                        minimum=4.0,
                        maximum=10.0,
                        value=7.0,
                        step=0.5,
                        label="Siren End Time (Seconds)",
                        info="Time when ambulance turns off siren"
                    )

                    custom_file = gr.File(
                        label="Or Upload Custom 10s Traffic+Siren Audio (.wav)",
                        file_types=[".wav"]
                    )
                    separate_btn = gr.Button("⚡ Detect Siren & Separate Audio (10s)", variant="primary", elem_classes=["btn-primary"])

                with gr.Column(scale=2):
                    gr.Markdown("### 🚦 2. Traffic Light Controller & Detection Status")
                    status_banner = gr.Markdown("Click **Detect Siren & Separate Audio** to analyze traffic stream.")
                    spec_plot = gr.Plot(label="Real-Time Detection Timeline & Spectrogram Dashboard")

            gr.Markdown("### 🔊 3. Full 10-Second Audio Track Playback")
            with gr.Row():
                audio_mix = gr.Audio(label="1. Mixed Input (Traffic + Siren Burst)", type="filepath")
                audio_separated = gr.Audio(label="2. Extracted Siren (Model Output)", type="filepath")
                audio_gt_siren = gr.Audio(label="3. Clean Ground-Truth Siren", type="filepath")
                audio_traffic = gr.Audio(label="4. Continuous Background Traffic", type="filepath")

            separate_btn.click(
                fn=process_audio_separation_and_detection,
                inputs=[siren_type, noise_type, snr_slider, siren_start_slider, siren_end_slider, custom_file],
                outputs=[audio_mix, audio_separated, audio_gt_siren, audio_traffic, spec_plot, status_banner]
            )

        with gr.Tab("📘 System Architecture & Real-Time Detection Guide"):
            gr.Markdown(
                """
                ### Real-Time Emergency Vehicle Traffic Signal System Architecture:
                1. **10-Second Audio Streaming**: Background traffic plays continuously. Emergency vehicle siren turns on mid-stream (e.g. 3.0s to 7.0s).
                2. **Sliding Window Detection Engine**: A 4.0-second sliding window with 0.5-second hop steps scans the stream in real time.
                3. **Siren Confidence Scoring**: Computes siren energy ratio per window step $C(t) = \\frac{\\|\\hat{|S|}_t\\|_F}{\\||X_{\\text{mix}}|_t\\|_F}$.
                4. **Traffic Light Priority Signal Override**: If confidence exceeds 18%, the system triggers an emergency override to change traffic signals to **GREEN**.
                5. **Overlap-Add (OLA) Synthesis**: Seamlessly reconstructs full 10-second isolated siren audio without boundary click artifacts.
                """
            )

    print("Launching upgraded Real-Time Detection Gradio App...")
    demo.launch(share=False)


if __name__ == "__main__":
    launch_app()
