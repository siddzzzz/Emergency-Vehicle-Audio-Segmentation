# Emergency Vehicle Audio Segmentation & Source Separation

A self-contained, lightweight PyTorch project designed to isolate emergency vehicle sirens (ambulances, police cars, firetrucks) from heavy background traffic noise using STFT Spectrogram Ratio Masking with a 2D U-Net.

---

## Key Features

1. **Self-Contained & Low Memory**: Uses an **On-The-Fly Audio Mixer** and procedural audio synthesis engine. Requires **< 50MB** of disk space without downloading gigabytes of heavy datasets.
2. **100% Portable (Relative Paths)**: All file paths for data loading, checkpoints, and model weights are strictly relative. You can zip or move this directory anywhere and run immediately.
3. **RTX 3050 GPU Ready**: Optimized for fast training with PyTorch (`device = 'cuda'`), lightweight memory consumption (< 3GB VRAM), and SI-SDR loss metrics.
4. **Interactive Gradio Web Dashboard**: Includes visual STFT Spectrogram comparison plots and interactive audio players to listen to mixed inputs vs. separated siren outputs.

---

## Directory Structure

```
Emergency-Vehicle-Audio-Segmentation/
│
├── data/                    # Generated/placed audio samples (relative path)
│   ├── sirens/              # Clean siren WAV clips
│   └── traffic/             # Background traffic noise WAV clips
│
├── checkpoints/             # Saved PyTorch model checkpoints
│   └── best_model.pth
│
├── src/                     # Core Python modules
│   ├── data_generator.py    # Procedural siren & traffic sound synthesizer
│   ├── dataset.py           # On-the-fly PyTorch DataLoader with dynamic SNR mixing
│   ├── model.py             # 2D Spectrogram U-Net Ratio Masking Architecture
│   └── metrics.py           # SI-SDR metric & audio separation loss functions
│
├── train.py                 # Training script with CLI flags and validation
├── app.py                   # Interactive Gradio Audio Web Application
├── requirements.txt         # Project python dependencies
└── README.md                # Project documentation
```

---

## Quickstart & Setup

### 1. Install Dependencies

In your Python / CUDA environment, install the required packages:

```bash
pip install -r requirements.txt
```

*(Note: For GPU acceleration on your RTX 3050, ensure PyTorch with CUDA is installed: `pip install torch --index-url https://download.pytorch.org/whl/cu124`)*

---

### 2. Generate Initial Sample Dataset (Optional)

Generate sample siren and traffic audio clips in `data/sirens` and `data/traffic`:

```bash
python src/data_generator.py
```

---

### 3. Dry-Run Verification

Test the data loader, STFT transform, model forward/backward pass, and checkpoint saving with a quick 2-step verification:

```bash
python train.py --dry-run
```

---

### 4. Train the Model (In your CUDA Environment)

To launch a full training run on your GPU:

```bash
python train.py --epochs 30 --batch-size 16 --lr 0.001
```

The script will save the best model weights to `checkpoints/best_model.pth`.

---

### 5. Launch Interactive Audio Web Dashboard

Run the Gradio web application to test audio separation interactively:

```bash
python app.py
```

- Adjust the **Signal-to-Noise Ratio (SNR in dB)** slider.
- Select different siren types (Wail, Yelp, Hi-Lo) or upload custom traffic audio.
- Click **Separate Emergency Siren** to visualize STFT Spectrograms and play the isolated emergency vehicle audio!