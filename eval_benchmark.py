import os
import time
import json
from pathlib import Path
import numpy as np
from tqdm import tqdm

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
from src.metrics import calculate_si_sdr

BASE_DIR = Path(__file__).resolve().parent
CKPT_PATH = BASE_DIR / "checkpoints" / "best_model.pth"
LOG_DIR = BASE_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)

SAMPLE_RATE = 16000
duration = 10.0


def run_benchmark(num_trials=20):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"==================================================")
    print(f"--- Running Emergency Vehicle AI Benchmark Suite ---")
    print(f"Device: {device.upper()} | Samples: {num_trials}")
    print(f"==================================================")

    processor = AudioProcessor(sample_rate=SAMPLE_RATE)
    model = SpectrogramUNet().to(device)
    model.eval()

    if CKPT_PATH.exists():
        try:
            ckpt = torch.load(CKPT_PATH, map_location=device)
            model.load_state_dict(ckpt["model_state_dict"])
            print(f"Loaded trained model weights from {CKPT_PATH}")
        except Exception as e:
            print(f"Notice: Could not load checkpoint ({e}). Running with default initialized weights.")
    else:
        print("Notice: Running with default initialized model weights.")

    detector = SlidingWindowDetector(model, processor, window_sec=4.0, hop_sec=0.5, sample_rate=SAMPLE_RATE)
    spatial_engine = SpatialAudioEngine(mic_distance=0.5, sample_rate=SAMPLE_RATE)

    siren_gens = [
        ("Ambulance (Wail)", generate_siren_wail, 1),
        ("Police (Yelp)", generate_siren_yelp, 2),
        ("Firetruck (Hi-Lo)", generate_siren_hilo, 3)
    ]
    noise_types = ["rumble", "horns", "rain", "wind", "brakes", "construction"]
    snr_levels = [-10.0, -5.0, 0.0, 5.0, 10.0]
    test_angles = [-45.0, -25.0, 0.0, 25.0, 45.0]

    sdr_improvements = []
    angle_errors = []
    cls_correct = 0
    total_proc_time = 0.0

    print("\nRunning automated evaluation trials...")
    for idx in tqdm(range(num_trials), desc="Benchmarking"):
        s_name, s_gen, target_class = siren_gens[idx % len(siren_gens)]
        n_type = noise_types[idx % len(noise_types)]
        snr = snr_levels[idx % len(snr_levels)]
        target_angle = test_angles[idx % len(test_angles)]

        # Generate audio mixture
        siren_np = s_gen(duration_sec=duration, sample_rate=SAMPLE_RATE, start_sec=2.0, end_sec=7.0, vehicle_speed_kmh=60.0)
        traffic_np = generate_traffic_noise(duration_sec=duration, sample_rate=SAMPLE_RATE, noise_type=n_type)

        siren_tensor = torch.tensor(siren_np, dtype=torch.float32)
        traffic_tensor = torch.tensor(traffic_np, dtype=torch.float32)
        traffic_tensor = (traffic_tensor / (torch.max(torch.abs(traffic_tensor)) + 1e-6)) * 0.45

        # Active siren SNR scaling
        active_mask = torch.abs(siren_tensor) > 1e-4
        if torch.any(active_mask):
            p_siren = torch.mean(siren_tensor[active_mask] ** 2) + 1e-8
        else:
            p_siren = torch.mean(siren_tensor ** 2) + 1e-8

        p_traffic = torch.mean(traffic_tensor ** 2) + 1e-8
        scale = torch.sqrt(p_traffic * (10.0 ** (snr / 10.0)) / p_siren)
        clean_siren_wave = siren_tensor * scale

        # Simulate 2-mic spatial stereo mixture
        m1_np, m2_np, _, _ = spatial_engine.simulate_stereo_mixture(
            clean_siren_wave.numpy(),
            traffic_tensor.numpy(),
            angle_deg=target_angle
        )

        m1_tensor = torch.tensor(m1_np, dtype=torch.float32)
        m2_tensor = torch.tensor(m2_np, dtype=torch.float32)

        # Measure Processing Time
        t_start = time.time()
        res = detector.process_stream(m1_tensor, mic2_waveform=m2_tensor, spatial_engine=spatial_engine, device=device)
        t_elapsed = time.time() - t_start
        total_proc_time += t_elapsed

        # 1. Evaluate SI-SDR Improvement
        est_wave = res["full_separated_wave"]
        in_sdr = calculate_si_sdr(m1_tensor, clean_siren_wave).item()
        out_sdr = calculate_si_sdr(est_wave, clean_siren_wave).item()
        sdr_gain = out_sdr - in_sdr
        sdr_improvements.append(sdr_gain)

        # 2. Evaluate DoA Angle Estimation Error
        est_angle = res["estimated_angle"]
        angle_err = abs(target_angle - est_angle)
        angle_errors.append(angle_err)

        # 3. Evaluate Vehicle Classification
        pred_vehicle = res["top_vehicle_name"]
        if s_name.split()[0] in pred_vehicle:
            cls_correct += 1

    mean_sdr_gain = float(np.mean(sdr_improvements))
    mean_angle_err = float(np.mean(angle_errors))
    cls_accuracy = float(cls_correct / num_trials) * 100.0
    avg_time_per_10s = float(total_proc_time / num_trials)
    real_time_factor = float(duration / avg_time_per_10s)

    # Benchmark Results Summary Report
    report = {
        "device": device,
        "total_trials": num_trials,
        "mean_sdr_improvement_db": round(mean_sdr_gain, 2),
        "classification_accuracy_pct": round(cls_accuracy, 1),
        "doa_mean_absolute_angle_error_deg": round(mean_angle_err, 2),
        "avg_processing_time_per_10s_stream_sec": round(avg_time_per_10s, 3),
        "real_time_processing_factor_rtf": round(real_time_processing_factor_rtf if 'real_time_processing_factor_rtf' in locals() else real_time_factor, 1)
    }

    print("\n==================================================")
    print("--- AUTOMATED BENCHMARK EVALUATION RESULTS ---")
    print("==================================================")
    print(f"- Mean SI-SDR Improvement:    +{mean_sdr_gain:.2f} dB")
    print(f"- Vehicle Classification Acc: {cls_accuracy:.1f} %")
    print(f"- DoA Angle Error:           {mean_angle_err:.2f} deg")
    print(f"- Avg 10s Processing Time:    {avg_time_per_10s:.3f} s ({real_time_factor:.1f}x Real-Time Speed)")
    print("==================================================\n")

    # Save benchmark JSON file
    res_path = LOG_DIR / "benchmark_results.json"
    with open(res_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"Benchmark results saved to: {res_path}")
    return report


if __name__ == "__main__":
    run_benchmark(num_trials=15)
