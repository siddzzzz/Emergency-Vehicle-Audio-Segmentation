import os
import argparse
from pathlib import Path
from tqdm import tqdm

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from src.dataset import EmergencyAudioDataset
from src.model import SpectrogramUNet
from src.metrics import AudioSeparationLoss, calculate_si_sdr

# Relative paths base
BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = BASE_DIR / "data"
DEFAULT_CKPT_DIR = BASE_DIR / "checkpoints"


def train_model(
    epochs=10,
    batch_size=8,
    lr=1e-3,
    data_dir=None,
    checkpoint_dir=None,
    dry_run=False,
    device=None
):
    if data_dir is None:
        data_dir = DEFAULT_DATA_DIR
    else:
        data_dir = Path(data_dir)

    if checkpoint_dir is None:
        checkpoint_dir = DEFAULT_CKPT_DIR
    else:
        checkpoint_dir = Path(checkpoint_dir)

    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"--- Starting Emergency Vehicle Audio Separation Training ---")
    print(f"Device: {device}")
    print(f"Data Dir: {data_dir}")
    print(f"Checkpoint Dir: {checkpoint_dir}")
    print(f"Dry Run Mode: {dry_run}")

    # Prepare datasets and loaders
    train_dataset = EmergencyAudioDataset(
        data_dir=data_dir,
        dataset_size=20 if dry_run else 400
    )
    val_dataset = EmergencyAudioDataset(
        data_dir=data_dir,
        dataset_size=5 if dry_run else 50
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Initialize Model, Loss, Optimizer
    model = SpectrogramUNet().to(device)
    criterion = AudioSeparationLoss(spectral_weight=1.0, wave_weight=0.05)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)

    best_val_loss = float("inf")
    processor = train_dataset.processor

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        
        loop = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} [Train]")
        for step, batch in enumerate(loop):
            mix_mag = batch["mix_mag"].to(device)
            siren_mag = batch["siren_mag"].to(device)
            class_id = batch["class_id"].to(device)

            optimizer.zero_grad()
            mask, est_mag, class_logits = model(mix_mag)

            loss = criterion(est_mag, siren_mag, class_logits=class_logits, target_class=class_id)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            loop.set_postfix(loss=loss.item())

            if dry_run and step >= 1:
                print("Dry-run limit reached for training step.")
                break

        train_loss /= len(train_loader)

        # Validation phase
        model.eval()
        val_loss = 0.0
        total_sdr = 0.0
        val_steps = 0

        with torch.no_grad():
            for step, batch in enumerate(val_loader):
                mix_mag = batch["mix_mag"].to(device)
                mix_phase = batch["mix_phase"].to(device)
                siren_mag = batch["siren_mag"].to(device)
                siren_wave = batch["siren_wave"].to(device)
                class_id = batch["class_id"].to(device)

                mask, est_mag, class_logits = model(mix_mag)
                loss = criterion(est_mag, siren_mag, class_logits=class_logits, target_class=class_id)
                val_loss += loss.item()

                # Reconstruct estimated wave to evaluate SI-SDR metric
                est_wave = processor.istft(est_mag[0], mix_phase[0])
                sdr_val = calculate_si_sdr(est_wave, siren_wave[0])
                total_sdr += sdr_val.item()
                val_steps += 1

                if dry_run and step >= 1:
                    print("Dry-run limit reached for validation step.")
                    break

        val_loss /= max(val_steps, 1)
        avg_sdr = total_sdr / max(val_steps, 1)

        print(f"Epoch {epoch:02d}/{epochs:02d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val SI-SDR: {avg_sdr:.2f} dB")
        scheduler.step(val_loss)

        # Save Best Model Checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            ckpt_path = checkpoint_dir / "best_model.pth"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "val_sdr": avg_sdr
            }, ckpt_path)
            print(f" Saved checkpoint to {ckpt_path}")

        if dry_run:
            print("Dry run test completed successfully!")
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Emergency Vehicle Audio Separation Model")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--data-dir", type=str, default=None, help="Relative or absolute path to data folder")
    parser.add_argument("--checkpoint-dir", type=str, default=None, help="Relative or absolute path to checkpoint folder")
    parser.add_argument("--dry-run", action="store_true", help="Run 1-2 steps for verification testing only")

    args = parser.parse_args()
    train_model(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        data_dir=args.data_dir,
        checkpoint_dir=args.checkpoint_dir,
        dry_run=args.dry_run
    )
