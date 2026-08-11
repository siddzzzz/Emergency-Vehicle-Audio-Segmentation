import os
import json
from datetime import datetime
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
LOG_DIR = BASE_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)


class TrafficEventLogger:
    """
    Traffic Event Logger:
    Generates and exports structured JSON logs for smart city traffic control systems,
    recording emergency vehicle detections, DoA angles, target lanes, and AI confidence.
    """
    def __init__(self, log_dir=None):
        self.log_dir = Path(log_dir) if log_dir else LOG_DIR
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def log_event(self, detection_res, siren_type, noise_type, snr_db, vehicle_speed_kmh, angle_deg):
        """
        Creates a structured traffic event log dictionary and appends it to logs/traffic_event_log.json.
        """
        timestamp_str = datetime.now().isoformat(timespec="seconds")
        event_id = f"event_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        log_entry = {
            "event_id": event_id,
            "timestamp": timestamp_str,
            "stream_duration_sec": 10.0,
            "siren_detected": len(detection_res["detected_intervals"]) > 0,
            "vehicle_type": detection_res["top_vehicle_name"],
            "detected_intervals_sec": detection_res["detected_intervals"],
            "target_angle_deg": angle_deg,
            "estimated_angle_deg": detection_res["estimated_angle"],
            "angle_error_deg": round(abs(angle_deg - detection_res["estimated_angle"]), 1),
            "target_traffic_lane": detection_res["target_lane"],
            "traffic_signal_action": detection_res["traffic_action"],
            "snr_db": snr_db,
            "vehicle_speed_kmh": vehicle_speed_kmh,
            "siren_type_input": siren_type,
            "noise_environment": noise_type
        }

        # Append to main log file
        log_file = self.log_dir / "traffic_event_log.json"
        existing_logs = []
        if log_file.exists():
            try:
                with open(log_file, "r", encoding="utf-8") as f:
                    existing_logs = json.load(f)
            except Exception:
                existing_logs = []

        existing_logs.append(log_entry)

        with open(log_file, "w", encoding="utf-8") as f:
            json.dump(existing_logs, f, indent=2)

        # Save latest event single file for Gradio download
        latest_file = self.log_dir / "latest_event.json"
        with open(latest_file, "w", encoding="utf-8") as f:
            json.dump(log_entry, f, indent=2)

        return log_entry, str(latest_file)
