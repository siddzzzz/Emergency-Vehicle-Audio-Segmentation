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
from src.spatial_audio import SpatialAudioEngine

# Relative directory paths
BASE_DIR = Path(__file__).resolve().parent
CKPT_PATH = BASE_DIR / "checkpoints" / "best_model.pth"
TEMP_DIR = BASE_DIR / "temp_audio"
TEMP_DIR.mkdir(exist_ok=True)

# Initialize Processor, Model, and Spatial Audio Engine
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

# Initialize Sliding Window Detector & Spatial DoA Engine
detector = SlidingWindowDetector(model, processor, window_sec=4.0, hop_sec=0.5, sample_rate=SAMPLE_RATE)
spatial_engine = SpatialAudioEngine(mic_distance=0.5, sample_rate=SAMPLE_RATE)


def plot_detection_dashboard(mix_wave_np, time_points, confidence_curve, class_probs_history, detected_intervals, estimated_angle, target_lane, mix_mag, mask, siren_pred_mag, clean_siren_mag=None):
    """
    Generates a 6-Subplot Visualizer Dashboard:
    1. 1D Mixed Waveform Timeline with Red Highlighted Emergency Siren Intervals
    2. Multi-Class Emergency Vehicle Classification Probabilities (%) over Time
    3. 2D Traffic Junction Polar Radar Plot (Microphone Array & Vehicle DoA Angle)
    4. Mixed Input Spectrogram
    5. Predicted U-Net Ratio Mask M(f, t)
    6. Extracted Siren Spectrogram
    """
    fig = plt.figure(figsize=(16, 9.5), dpi=120)
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.2])

    t_axis = np.linspace(0, len(mix_wave_np) / SAMPLE_RATE, len(mix_wave_np))

    # --- Subplot 1: 1D Mixed Audio Waveform Timeline ---
    ax_wave = fig.add_subplot(gs[0, 0])
    ax_wave.plot(t_axis, mix_wave_np, color="#4b5563", alpha=0.7, label="Traffic + Siren Audio")
    
    for start_t, end_t in detected_intervals:
        ax_wave.axvspan(start_t, end_t, color="#ef4444", alpha=0.3, label="Siren Active" if start_t == detected_intervals[0][0] else "")
    
    ax_wave.set_title("1. Mixed Audio Waveform (10s)", fontsize=10, fontweight="bold")
    ax_wave.set_xlabel("Time (s)")
    ax_wave.set_ylabel("Amplitude")
    ax_wave.set_xlim(0, len(mix_wave_np) / SAMPLE_RATE)
    ax_wave.grid(True, linestyle="--", alpha=0.5)
    ax_wave.legend(loc="upper right", fontsize=8)

    # --- Subplot 2: Multi-Class Vehicle Classifier Probabilities ---
    ax_conf = fig.add_subplot(gs[0, 1])
    if class_probs_history is not None and len(class_probs_history) > 0:
        probs_pct = class_probs_history * 100.0
        ax_conf.plot(time_points, probs_pct[:, 1], color="#dc2626", linewidth=1.8, marker="o", markersize=3, label="Ambulance (Wail)")
        ax_conf.plot(time_points, probs_pct[:, 2], color="#2563eb", linewidth=1.8, marker="s", markersize=3, label="Police (Yelp)")
        ax_conf.plot(time_points, probs_pct[:, 3], color="#d97706", linewidth=1.8, marker="^", markersize=3, label="Firetruck (Hi-Lo)")
        ax_conf.plot(time_points, probs_pct[:, 0], color="#6b7280", linewidth=1.2, linestyle="--", label="Traffic Only")
    else:
        ax_conf.plot(time_points, confidence_curve, color="#dc2626", linewidth=2.0, label="Siren Confidence")
    
    ax_conf.axhline(detector.detection_threshold * 100.0, color="#f59e0b", linestyle=":", linewidth=1.2, label="Threshold (18%)")
    ax_conf.set_title("2. Vehicle Class Probabilities (%)", fontsize=10, fontweight="bold")
    ax_conf.set_xlabel("Time (s)")
    ax_conf.set_ylabel("Probability (%)")
    ax_conf.set_ylim(0, 105)
    ax_conf.set_xlim(0, len(mix_wave_np) / SAMPLE_RATE)
    ax_conf.grid(True, linestyle="--", alpha=0.5)
    ax_conf.legend(loc="upper right", fontsize=7)

    # --- Subplot 3: 2D Traffic Junction DoA Polar Radar Plot ---
    ax_radar = fig.add_subplot(gs[0, 2], polar=True)
    ax_radar.set_theta_zero_location("N")
    ax_radar.set_theta_direction(-1)  # Clockwise
    
    # Angles for 4 Lanes
    angles_rad = np.radians([0, 45, -45])
    labels = ["Lane 1\n(North)", "Lane 2\n(East)", "Lane 3\n(West)"]
    
    # Plot microphone array at center
    ax_radar.plot(0, 0, marker="X", color="#10b981", markersize=10, label="Mic Array (Junction)")

    # Highlight estimated vehicle angle if siren detected
    if detected_intervals:
        est_rad = np.radians(estimated_angle)
        ax_radar.annotate(
            f"EMERGENCY VEHICLE\n({estimated_angle:+.1f}°)",
            xy=(est_rad, 0.85),
            xytext=(est_rad, 1.15),
            arrowprops=dict(facecolor="#ef4444", shrink=0.05, width=2, headwidth=8),
            fontsize=8,
            fontweight="bold",
            color="#ef4444",
            ha="center"
        )
        ax_radar.plot([0, est_rad], [0, 0.85], color="#ef4444", linewidth=2.5, linestyle="-")

    ax_radar.set_title(f"3. DoA Radar: {target_lane}", fontsize=10, fontweight="bold", pad=12)
    ax_radar.set_rmax(1.0)
    ax_radar.grid(True, linestyle="--", alpha=0.5)

    # --- Subplots 4, 5, 6: Spectrograms & Learned Ratio Mask ---
    eps = 1e-6
    mix_db = 20 * np.log10(mix_mag + eps)
    pred_db = 20 * np.log10(siren_pred_mag + eps)

    # 4. Mixed Spectrogram
    ax_spec1 = fig.add_subplot(gs[1, 0])
    im0 = ax_spec1.imshow(mix_db, aspect="auto", origin="lower", cmap="inferno")
    ax_spec1.set_title("4. Mixed Input Spectrogram", fontsize=10, fontweight="bold")
    ax_spec1.set_xlabel("Time Frames")
    ax_spec1.set_ylabel("Frequency Bins")
    fig.colorbar(im0, ax=ax_spec1, format="%+2.0f dB")

    # 5. Predicted Ratio Mask M(f,t)
    ax_mask = fig.add_subplot(gs[1, 1])
    im1 = ax_mask.imshow(mask, aspect="auto", origin="lower", cmap="viridis", vmin=0.0, vmax=1.0)
    ax_mask.set_title("5. Predicted Ratio Mask M(f,t)", fontsize=10, fontweight="bold")
    ax_mask.set_xlabel("Time Frames")
    ax_mask.set_ylabel("Frequency Bins")
    cbar1 = fig.colorbar(im1, ax=ax_mask)
    cbar1.set_label("Mask Value (0=Noise, 1=Siren)", fontsize=8)

    # 6. Extracted Siren Spectrogram
    ax_spec2 = fig.add_subplot(gs[1, 2])
    im2 = ax_spec2.imshow(pred_db, aspect="auto", origin="lower", cmap="inferno")
    ax_spec2.set_title("6. Extracted Siren Spectrogram", fontsize=10, fontweight="bold")
    ax_spec2.set_xlabel("Time Frames")
    ax_spec2.set_ylabel("Frequency Bins")
    fig.colorbar(im2, ax=ax_spec2, format="%+2.0f dB")

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


def process_audio_separation_and_detection(siren_type, noise_type, snr_db, start_sec, end_sec, angle_deg, vehicle_speed_kmh=60.0, closest_approach_sec=5.0, custom_audio_file=None):
    """
    Gradio Event Handler:
    1. Synthesizes siren burst + Doppler pitch shift + distance volume gain + background traffic noise.
    2. Simulates 2-microphone array spatial stereo audio at traffic junction for target angle.
    3. Runs Sliding Window AI Detector + GCC-PHAT DoA Angle Estimator.
    4. Emits lane-specific traffic light priority signal override.
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

        mix_wave_mic1 = torch.tensor(audio_np, dtype=torch.float32)
        mix_wave_mic2 = mix_wave_mic1
    else:
        # Generate 10-second siren burst with Doppler Effect
        if siren_type == "Wail (Ambulance)":
            siren_np = generate_siren_wail(
                duration_sec=duration, sample_rate=SAMPLE_RATE, start_sec=start_sec, end_sec=end_sec,
                vehicle_speed_kmh=vehicle_speed_kmh, closest_approach_sec=closest_approach_sec
            )
        elif siren_type == "Yelp (Police)":
            siren_np = generate_siren_yelp(
                duration_sec=duration, sample_rate=SAMPLE_RATE, start_sec=start_sec, end_sec=end_sec,
                vehicle_speed_kmh=vehicle_speed_kmh, closest_approach_sec=closest_approach_sec
            )
        else:
            siren_np = generate_siren_hilo(
                duration_sec=duration, sample_rate=SAMPLE_RATE, start_sec=start_sec, end_sec=end_sec,
                vehicle_speed_kmh=vehicle_speed_kmh, closest_approach_sec=closest_approach_sec
            )

        # Map UI noise choice to generator noise type key
        if noise_type == "Heavy Rain Storm":
            n_type = "rain"
        elif noise_type == "High Wind Gusts":
            n_type = "wind"
        elif noise_type == "Diesel Truck Air Brakes":
            n_type = "brakes"
        elif noise_type == "Urban Construction Site":
            n_type = "construction"
        elif noise_type == "Harsh Storm (Rain + Wind + Traffic)":
            n_type = "storm"
        elif noise_type == "Traffic + Horns":
            n_type = "horns"
        else:
            n_type = "rumble"

        # Generate 10-second background environmental noise
        traffic_np = generate_traffic_noise(
            duration_sec=duration,
            sample_rate=SAMPLE_RATE,
            noise_type=n_type
        )

        siren_tensor = torch.tensor(siren_np, dtype=torch.float32)
        traffic_tensor = torch.tensor(traffic_np, dtype=torch.float32)

        # Normalize traffic baseline
        traffic_tensor = (traffic_tensor / (torch.max(torch.abs(traffic_tensor)) + 1e-6)) * 0.45

        # Calculate siren power over active frames only
        active_mask = torch.abs(siren_tensor) > 1e-4
        if torch.any(active_mask):
            p_siren = torch.mean(siren_tensor[active_mask] ** 2) + 1e-8
        else:
            p_siren = torch.mean(siren_tensor ** 2) + 1e-8

        p_traffic = torch.mean(traffic_tensor ** 2) + 1e-8
        target_siren_power = p_traffic * (10.0 ** (snr_db / 10.0))
        scale = torch.sqrt(target_siren_power / p_siren)

        clean_siren_wave = siren_tensor * scale
        traffic_wave = traffic_tensor

        # Simulate 2-Microphone Array Spatial Audio for specified target angle
        mic1_np, mic2_np, _, _ = spatial_engine.simulate_stereo_mixture(
            clean_siren_wave.numpy(),
            traffic_wave.numpy(),
            angle_deg=angle_deg
        )

        mix_wave_mic1 = torch.tensor(mic1_np, dtype=torch.float32)
        mix_wave_mic2 = torch.tensor(mic2_np, dtype=torch.float32)

        # Soft clip mixture to prevent digital clipping
        if torch.max(torch.abs(mix_wave_mic1)) > 1.0:
            mix_wave_mic1 = torch.tanh(mix_wave_mic1)
        if torch.max(torch.abs(mix_wave_mic2)) > 1.0:
            mix_wave_mic2 = torch.tanh(mix_wave_mic2)

    # Real-Time Sliding Window Detector + GCC-PHAT Spatial DoA Engine
    detection_res = detector.process_stream(
        mix_wave_mic1,
        mic2_waveform=mix_wave_mic2,
        spatial_engine=spatial_engine,
        device=device
    )
    
    full_est_siren_wave = detection_res["full_separated_wave"]
    time_points = detection_res["time_points"]
    confidence_curve = detection_res["confidence_curve"]
    class_probs_history = detection_res["class_probs_history"]
    detected_intervals = detection_res["detected_intervals"]
    estimated_angle = detection_res["estimated_angle"]
    target_lane = detection_res["target_lane"]
    traffic_action_msg = detection_res["traffic_action"]

    mix_np = mix_wave_mic1.cpu().numpy()
    est_siren_np = full_est_siren_wave.cpu().numpy()

    # Save audio files for Gradio players
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
    mix_mag, _ = processor.stft(mix_wave_mic1)
    mix_mag_input = mix_mag.unsqueeze(0).to(device)

    with torch.no_grad():
        mask_tensor, pred_mag_tensor, class_logits = model(mix_mag_input)

    mask_np = mask_tensor.squeeze().cpu().numpy()
    pred_mag_np = pred_mag_tensor.squeeze().cpu().numpy()
    mix_mag_np = mix_mag.squeeze().cpu().numpy()

    fig = plot_detection_dashboard(
        mix_np,
        time_points,
        confidence_curve,
        class_probs_history,
        detected_intervals,
        estimated_angle,
        target_lane,
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
    .main-container { max-width: 1400px; margin: 0 auto; }
    .btn-primary { background: linear-gradient(135deg, #ef4444 0%, #b91c1c 100%) !important; color: white !important; font-weight: bold !important; font-size: 16px !important; }
    .traffic-box { border: 2px solid #ef4444; border-radius: 8px; padding: 16px; background-color: #111827; }
    """

    theme = gr.themes.Soft(primary_hue="red", secondary_hue="slate")
    with gr.Blocks(theme=theme, css=custom_css, title="Emergency Vehicle DoA Audio Detection & Separation") as demo:
        gr.Markdown(
            """
            # 🚨 Spatial Audio Emergency Vehicle Detection & Traffic Light Priority Controller
            *Isolate sirens, classify vehicle types, and estimate Direction of Arrival (DoA) angle using GCC-PHAT for targeted traffic light lane overrides.*
            """
        )

        with gr.Tab("🚦 Traffic Junction Spatial Detector & Separator"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### 🎛️ 1. Audio Stream & Spatial Controls")
                    siren_type = gr.Dropdown(
                        choices=["Wail (Ambulance)", "Yelp (Police)", "Hi-Lo (Firetruck)"],
                        value="Wail (Ambulance)",
                        label="Emergency Siren Type"
                    )
                    noise_type = gr.Dropdown(
                        choices=[
                            "Heavy Traffic Rumble",
                            "Traffic + Horns",
                            "Heavy Rain Storm",
                            "High Wind Gusts",
                            "Diesel Truck Air Brakes",
                            "Urban Construction Site",
                            "Harsh Storm (Rain + Wind + Traffic)"
                        ],
                        value="Traffic + Horns",
                        label="Background Acoustic Environment"
                    )
                    snr_slider = gr.Slider(
                        minimum=-12.0,
                        maximum=12.0,
                        value=0.0,
                        step=1.0,
                        label="Signal-to-Noise Ratio (SNR in dB)",
                        info="Lower dB = Traffic is louder than siren"
                    )
                    
                    gr.Markdown("#### 🏎️ 2. Doppler Pitch Shift & Physical Dynamics")
                    speed_slider = gr.Slider(
                        minimum=0.0,
                        maximum=100.0,
                        value=60.0,
                        step=5.0,
                        label="Vehicle Approaching Speed (km/h)",
                        info="0 = Stationary | 60 km/h = Pitch shifts higher when approaching and drops as it passes"
                    )
                    closest_time_slider = gr.Slider(
                        minimum=1.0,
                        maximum=9.0,
                        value=5.0,
                        step=0.5,
                        label="Closest Approach Time (Seconds)",
                        info="Time when vehicle passes closest to traffic junction"
                    )

                    gr.Markdown("#### 📍 3. Approaching Vehicle Angle & Lane Direction")
                    angle_slider = gr.Slider(
                        minimum=-60.0,
                        maximum=60.0,
                        value=45.0,
                        step=5.0,
                        label="Approaching Vehicle Angle (Degrees)",
                        info="0° = Lane 1 (North), +45° = Lane 2 (East Right), -45° = Lane 3 (West Left)"
                    )

                    gr.Markdown("#### ⏱️ 4. Siren Active Burst Window")
                    siren_start_slider = gr.Slider(
                        minimum=0.0,
                        maximum=6.0,
                        value=3.0,
                        step=0.5,
                        label="Siren Start Time (Seconds)",
                        info="Time when siren turns on"
                    )
                    siren_end_slider = gr.Slider(
                        minimum=4.0,
                        maximum=10.0,
                        value=7.0,
                        step=0.5,
                        label="Siren End Time (Seconds)",
                        info="Time when siren turns off"
                    )

                    custom_file = gr.File(
                        label="Or Upload Custom Traffic+Siren Audio (.wav)",
                        file_types=[".wav"]
                    )
                    separate_btn = gr.Button("⚡ Detect Siren, Lane & Separate Audio (10s)", variant="primary", elem_classes=["btn-primary"])

                with gr.Column(scale=2):
                    gr.Markdown("### 🚦 2. Traffic Light Controller & Spatial Junction Radar")
                    status_banner = gr.Markdown("Click **Detect Siren, Lane & Separate Audio** to analyze traffic junction.")
                    spec_plot = gr.Plot(label="Spatial DoA Radar & Spectrogram Dashboard")

            gr.Markdown("### 🔊 3. Full 10-Second Audio Track Playback")
            with gr.Row():
                audio_mix = gr.Audio(label="1. Mixed Input (Mic 1 Junction Stream)", type="filepath")
                audio_separated = gr.Audio(label="2. Extracted Siren (Model Output)", type="filepath")
                audio_gt_siren = gr.Audio(label="3. Clean Ground-Truth Siren", type="filepath")
                audio_traffic = gr.Audio(label="4. Continuous Background Traffic", type="filepath")

            separate_btn.click(
                fn=process_audio_separation_and_detection,
                inputs=[siren_type, noise_type, snr_slider, siren_start_slider, siren_end_slider, angle_slider, speed_slider, closest_time_slider, custom_file],
                outputs=[audio_mix, audio_separated, audio_gt_siren, audio_traffic, spec_plot, status_banner]
            )

        with gr.Tab("📘 Spatial Audio & DoA Architecture Guide"):
            gr.Markdown(
                """
                ### Spatial Audio Direction of Arrival (DoA) Architecture:
                1. **2-Microphone Array at Junction**: Microphones separated by distance $d = 0.5\\text{m}$ capture traffic sounds.
                2. **Time Difference of Arrival (TDoA)**: Sound arriving at angle $\\theta$ reaches Mic 2 with time delay $\\Delta t = \\frac{d \\sin(\\theta)}{v_{\\text{sound}}}$.
                3. **GCC-PHAT Cross-Correlation**: Estimates sub-millisecond time delay $\\hat{\\tau}$ using Generalized Cross-Correlation with Phase Transform.
                4. **Targeted Traffic Lane Override**: Calculates angle $\\theta = \\arcsin\\left(\\frac{\\hat{\\tau} v}{d}\\right)$ and triggers a green light override for the **EXACT approaching traffic lane**!
                """
            )

    print("Launching upgraded Spatial Audio Gradio App...")
    demo.launch(share=False)


if __name__ == "__main__":
    launch_app()
