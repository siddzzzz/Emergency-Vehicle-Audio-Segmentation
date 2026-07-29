import torch
import torch.nn.functional as F
import numpy as np


class SlidingWindowDetector:
    """
    Real-Time Sliding Window Detector & Audio Separator:
    Processes arbitrary length audio streams, detects emergency vehicle sirens over time,
    reconstructs continuous separated waveforms via Overlap-Add (OLA), and triggers traffic light priority signals.
    """
    def __init__(self, model, processor, window_sec=4.0, hop_sec=0.5, sample_rate=16000, detection_threshold=0.18):
        self.model = model
        self.processor = processor
        self.window_sec = window_sec
        self.hop_sec = hop_sec
        self.sample_rate = sample_rate
        self.detection_threshold = detection_threshold
        
        self.window_size = int(window_sec * sample_rate)
        self.hop_size = int(hop_sec * sample_rate)

    def process_stream(self, mix_waveform, device="cpu"):
        """
        Args:
            mix_waveform: 1D PyTorch tensor or NumPy array of raw mixed audio waveform (any length)
            device: 'cpu' or 'cuda'
        Returns:
            dict containing:
                - full_separated_wave: Reconstructed 1D separated siren waveform
                - time_points: Array of timestamp centers in seconds
                - confidence_curve: Array of siren energy/confidence percentages (0 to 100%)
                - detected_intervals: List of (start_sec, end_sec) tuples
                - traffic_action: Traffic signal decision string
        """
        if isinstance(mix_waveform, np.ndarray):
            mix_waveform = torch.tensor(mix_waveform, dtype=torch.float32)

        total_samples = mix_waveform.size(0)
        
        # If total audio is shorter than window_size, pad it
        if total_samples < self.window_size:
            pad_len = self.window_size - total_samples
            mix_waveform = F.pad(mix_waveform, (0, pad_len))
            total_samples = mix_waveform.size(0)

        # OLA Accumulators
        reconstructed_wave = torch.zeros(total_samples, dtype=torch.float32)
        weight_accumulator = torch.zeros(total_samples, dtype=torch.float32)
        window_weight = torch.hann_window(self.window_size)

        CLASS_NAMES = ["Traffic Noise", "Ambulance (Wail)", "Police (Yelp)", "Firetruck (Hi-Lo)"]

        time_points = []
        confidence_curve = []
        class_probs_history = []  # Matrix of (num_hops, 4)

        num_hops = (total_samples - self.window_size) // self.hop_size + 1

        self.model.eval()
        with torch.no_grad():
            for i in range(num_hops):
                start_idx = i * self.hop_size
                end_idx = start_idx + self.window_size
                if end_idx > total_samples:
                    break

                win_audio = mix_waveform[start_idx:end_idx]
                win_mag, win_phase = self.processor.stft(win_audio)
                win_mag_input = win_mag.unsqueeze(0).to(device)

                mask, pred_mag, class_logits = self.model(win_mag_input)
                pred_mag_cpu = pred_mag.squeeze(0).cpu()
                win_mag_cpu = win_mag.cpu()

                # Multi-Class Probabilities via Softmax
                probs = F.softmax(class_logits, dim=-1).squeeze(0).cpu().numpy()
                class_probs_history.append(probs)

                # Calculate Siren Energy Confidence Score
                siren_energy = torch.norm(pred_mag_cpu)
                mix_energy = torch.norm(win_mag_cpu) + 1e-8
                confidence = (siren_energy / mix_energy).item()
                confidence = float(np.clip(confidence, 0.0, 1.0))

                mid_timestamp = round((start_idx + end_idx) / (2.0 * self.sample_rate), 2)
                time_points.append(mid_timestamp)
                confidence_curve.append(confidence * 100.0)

                # iSTFT Waveform Reconstruction
                est_win_wave = self.processor.istft(pred_mag_cpu, win_phase)
                
                if est_win_wave.size(0) < self.window_size:
                    est_win_wave = F.pad(est_win_wave, (0, self.window_size - est_win_wave.size(0)))
                elif est_win_wave.size(0) > self.window_size:
                    est_win_wave = est_win_wave[:self.window_size]

                reconstructed_wave[start_idx:end_idx] += est_win_wave * window_weight
                weight_accumulator[start_idx:end_idx] += window_weight

        # Normalize OLA audio
        weight_accumulator = torch.clamp(weight_accumulator, min=1e-6)
        full_separated_wave = reconstructed_wave / weight_accumulator
        full_separated_wave = full_separated_wave[:total_samples]

        # Extract detected siren time intervals
        detected_intervals = []
        is_detecting = False
        start_t = 0.0

        for t_sec, conf_pct in zip(time_points, confidence_curve):
            if conf_pct >= (self.detection_threshold * 100.0):
                if not is_detecting:
                    is_detecting = True
                    start_t = max(0.0, t_sec - self.window_sec / 4.0)
            else:
                if is_detecting:
                    is_detecting = False
                    end_t = min(total_samples / self.sample_rate, t_sec + self.window_sec / 4.0)
                    detected_intervals.append((round(start_t, 1), round(end_t, 1)))

        if is_detecting:
            end_t = round(total_samples / self.sample_rate, 1)
            detected_intervals.append((round(start_t, 1), end_t))

        class_probs_history = np.array(class_probs_history)  # (num_hops, 4)

        # Extract top predicted emergency vehicle class
        top_class_id = 0
        top_vehicle_name = "Traffic Noise"
        if detected_intervals:
            # Average probabilities over detected active window hops
            active_hop_indices = [i for i, c in enumerate(confidence_curve) if c >= (self.detection_threshold * 100.0)]
            if active_hop_indices:
                mean_active_probs = np.mean(class_probs_history[active_hop_indices], axis=0)
                # Ignore class 0 (Traffic) when siren is active
                siren_probs = mean_active_probs[1:]
                top_class_id = int(np.argmax(siren_probs)) + 1
                top_vehicle_name = CLASS_NAMES[top_class_id]

        # Traffic Signal Decision logic
        if detected_intervals:
            interval_str = ", ".join([f"{s}s–{e}s" for s, e in detected_intervals])
            traffic_action = f"[EMERGENCY PRIORITY ACTIVATED] Detected {top_vehicle_name} in interval(s) [{interval_str}]. OVERRIDE SIGNAL TO GREEN FOR APPROACHING {top_vehicle_name.upper()}!"
        else:
            traffic_action = "[NORMAL TRAFFIC FLOW] No emergency sirens detected. Maintain regular automated traffic light cycle."

        return {
            "full_separated_wave": full_separated_wave,
            "time_points": np.array(time_points),
            "confidence_curve": np.array(confidence_curve),
            "class_probs_history": class_probs_history,
            "top_vehicle_name": top_vehicle_name,
            "detected_intervals": detected_intervals,
            "traffic_action": traffic_action
        }
